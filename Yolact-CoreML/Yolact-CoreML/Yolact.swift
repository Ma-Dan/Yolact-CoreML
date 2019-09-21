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
        let classIndex: Int
        let score: Float
        let rect: CGRect
    }

    let model = yolact()

    public init() { }

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
        
        print(keep.count)

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
    
        var boxes = decode(locs: loc_keep, priors: priors_keep)
        
        for i in 0..<boxes.count {
            let rect = CGRect(x: CGFloat(boxes[i][0]), y: CGFloat(boxes[i][1]),
                              width: CGFloat(boxes[i][2]-boxes[i][0]), height: CGFloat(boxes[i][3]-boxes[i][1]))
            
            let classIndex = argmax(scores: conf_keep[i])

            let prediction = Prediction(classIndex: classIndex,
                                        score: conf_keep[i][classIndex],
                                        rect: rect)
            predictions.append(prediction)
        }
        
        predictions = nonMaxSuppression(boxes: predictions, limit: Yolact.maxBoundingBoxes, threshold: iouThreshold)

        return predictions
    }
}
