import Foundation
import UIKit
import CoreML
import Accelerate

class Yolact {
    public static let inputWidth = 550
    public static let inputHeight = 550
    public static let maxBoundingBoxes = 10
    
    let classes = UInt(80)

    // Tweak these values to get more or fewer predictions.
    let confidenceThreshold: Float = 0.5
    let iouThreshold: Float = 0.5
    
    let priors = makePriors()

    struct Prediction {
        let index: Int
        let classIndex: Int
        let score: Float
        let rect: CGRect
        let box: [Float]
        var mask: UIImage
    }
    
    public struct PixelData {
        var a: UInt8
        var r: UInt8
        var g: UInt8
        var b: UInt8
    }
    
    var colors: [PixelData] = []

    let model = yolact()

    public init() {
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
          for g: CGFloat in [0.3, 0.7, 0.6, 0.8] {
            for b: CGFloat in [0.4, 0.8, 0.6, 1.0] {
                let color = PixelData(a: UInt8(255*0.8), r: UInt8(255*r), g: UInt8(255*g), b: UInt8(255*b))
              colors.append(color)
            }
          }
        }
    }

    public func predict(image: MLMultiArray) throws -> [Prediction] {
        if let output = try? model.prediction(_0: image) {
            //print("Predict", output._559.shape, output._755.shape, output._757.shape, output._758.shape)
            return computeMasks(features: [output._559, output._755, output._757, output._758])
        } else {
            return []
        }
    }
    
    public func decode(locs: [[Float]], priors: [[Float]]) -> [[Float]] {
        var boxes:[[Float]] = []
        
        for i in 0..<locs.count {
            var x1 = priors[i][0] + locs[i][0] * 0.1 * priors[i][2]
            var y1 = priors[i][1] + locs[i][1] * 0.1 * priors[i][3]
            var x2 = priors[i][2] * expf(locs[i][2] * 0.2)
            var y2 = priors[i][3] * expf(locs[i][3] * 0.2)
            
            x1 -= x2 / 2
            y1 -= y2 / 2
            x2 += x1
            y2 += y1
            
            boxes.append([x1, y1, x2, y2])
        }
        
        return boxes
    }
    
    func imageFromARGB32Bitmap(pixels: [PixelData], width: Int, height: Int) -> UIImage? {
        guard width > 0 && height > 0 else { return nil }
        guard pixels.count == width * height else { return nil }

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
        let bitsPerComponent = 8
        let bitsPerPixel = 32

        var data = pixels // Copy to mutable []
        guard let providerRef = CGDataProvider(data: NSData(bytes: &data,
                                length: data.count * MemoryLayout<PixelData>.size)
            )
            else { return nil }

        guard let cgim = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: width * MemoryLayout<PixelData>.size,
            space: rgbColorSpace,
            bitmapInfo: bitmapInfo,
            provider: providerRef,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
            )
            else { return nil }

        return UIImage(cgImage: cgim)
    }
    
    public func calcMask(proto: UnsafeMutablePointer<Float>, mask: UnsafeMutablePointer<Float>, index:Int, color: PixelData, box: [Float]) -> [PixelData] {
        var pixels = [PixelData]()
        
        for _ in 0..<138*138 {
            pixels.append(PixelData(a: 0, r: 0, g: 0, b: 0))
        }
        
        for y in 0..<138 {
            for x in 0..<138 {
                var sum: Float = 0
                for i in 0..<32 {
                    sum += proto[(y*138+x)*32+i] * mask[index*32+i]
                }
                sum = sigmoid(sum)
                
                if sum > 0.5 {
                    pixels[y*138+x] = color
                }
            }
        }
        
        let left = Int(138 * box[0])
        let top = Int(138 * box[1])
        let right = Int(138 * box[2])
        let bottom = Int(138 * box[3])
        
        for y in 0..<138 {
            for x in 0..<138 {
                if y < top || y > bottom {
                    pixels[y*138+x] = PixelData(a: 0, r: 0, g: 0, b: 0)
                    continue
                }
                if x < left || x > right {
                    pixels[y*138+x] = PixelData(a: 0, r: 0, g: 0, b: 0)
                }
            }
        }
        
        return pixels
    }

    public func computeMasks(features: [MLMultiArray]) -> [Prediction] {
        var predictions = [Prediction]()
    
        //Filter confidence with score
        var conf = UnsafeMutablePointer<Float>(OpaquePointer(features[3].dataPointer))
        conf = conf.advanced(by: 1)
    
        var keep:[Int] = []

        for i in 0..<19248 {
            var max: Float = 0
            vDSP_maxv(conf, 1, &max, classes)
        
            if max > confidenceThreshold {
                keep.append(i)
            }
        
            conf = conf.advanced(by: 81)
        }

        var loc_keep: [[Float]] = []
        var priors_keep: [[Float]] = []
        var conf_keep: [[Float]] = []
        let loc = UnsafeMutablePointer<Float>(OpaquePointer(features[1].dataPointer))
        let conf_read = UnsafeMutablePointer<Float>(OpaquePointer(features[3].dataPointer))
        for i in 0..<keep.count {
            loc_keep.append([loc[keep[i]*4], loc[keep[i]*4+1], loc[keep[i]*4+2], loc[keep[i]*4+3]])
            priors_keep.append(priors[keep[i]])
            
            var conf_one: [Float] = []
            for j in 0..<classes {
                conf_one.append(conf_read[keep[i]*81+1+Int(j)])
            }
            conf_keep.append(conf_one)
        }
    
        let boxes = decode(locs: loc_keep, priors: priors_keep)
        
        for i in 0..<boxes.count {
            let rect = CGRect(x: CGFloat(boxes[i][0]), y: CGFloat(boxes[i][1]),
                              width: CGFloat(boxes[i][2]-boxes[i][0]), height: CGFloat(boxes[i][3]-boxes[i][1]))
            
            let classIndex = argmax(scores: conf_keep[i])

            let prediction = Prediction(index: keep[i],
                                        classIndex: classIndex,
                                        score: conf_keep[i][classIndex],
                                        rect: rect,
                                        box: boxes[i],
                                        mask: UIImage())
            predictions.append(prediction)
        }
        
        predictions = nonMaxSuppression(boxes: predictions, limit: Yolact.maxBoundingBoxes, threshold: iouThreshold)
        
        //Calculate mask
        let proto = UnsafeMutablePointer<Float>(OpaquePointer(features[0].dataPointer))
        let mask = UnsafeMutablePointer<Float>(OpaquePointer(features[2].dataPointer))
        
        for i in 0..<predictions.count {
            let pixelData = calcMask(proto: proto, mask: mask, index: predictions[i].index, color: colors[predictions[i].classIndex], box: predictions[i].box)
            predictions[i].mask = imageFromARGB32Bitmap(pixels: pixelData, width: 138, height: 138) ?? UIImage()
        }

        return predictions
    }
}
