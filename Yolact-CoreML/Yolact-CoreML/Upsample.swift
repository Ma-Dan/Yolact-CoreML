//
//  Reshape.swift
//  Yolact-CoreML

import Foundation
import CoreML

@objc(Upsample) class Upsample: NSObject, MLCustomLayer {
    
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //print(#function, weights)
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        print(#function, inputShapes)
        
        let size = inputShapes[0][3].intValue
        
        var new_size = NSNumber(value: 0)
        if size == 18 {
            new_size = NSNumber(value: 35)
        }
        
        if size == 35 {
            new_size = NSNumber(value: 69)
        }
        
        if size == 69 {
            new_size = NSNumber(value: 138)
        }
        
        let outputShape = [[1, 1, inputShapes[0][2], new_size, new_size]]
        
        print("Upsample output shape", outputShape)

        return outputShape
    }
    
    func bilinear(src_mlma: MLMultiArray, dst_mlma: MLMultiArray, index: Int, src_w: Int, src_h: Int, dst_w: Int, dst_h: Int) {
        var src = UnsafeMutablePointer<Float>(OpaquePointer(src_mlma.dataPointer))
        src = src.advanced(by: index*src_w*src_h)
        var dst = UnsafeMutablePointer<Float>(OpaquePointer(dst_mlma.dataPointer))
        dst = dst.advanced(by: index*dst_w*dst_h)
        
        let h_ratio = (Float)(dst_h) / (Float)(src_h)
        let w_ratio = (Float)(dst_w) / (Float)(src_w)
        
        for y in 0..<dst_h {
            for x in 0..<dst_w {
                var px = (Int)((Float)(x) / w_ratio)
                var py = (Int)((Float)(y) / h_ratio)
                
                if px >= src_w - 1 {
                    px = src_w - 2
                }
                
                if py >= src_h - 1 {
                    py = src_h - 2
                }
                
                let fx1 = (Float)(x) / w_ratio - (Float)(px)
                let fx2 = 1.0 - fx1
                let fy1 = (Float)(y) / h_ratio - (Float)(py)
                let fy2 = 1.0 - fy1
                
                let w1 = fx2 * fy2
                let w2 = fx1 * fy2
                let w3 = fx2 * fy1
                let w4 = fx1 * fy1
                
                let p1 = src[py*src_w + px]
                let p2 = src[py*src_w + px + 1]
                let p3 = src[(py + 1)*src_w + px]
                let p4 = src[(py + 1)*src_w + px + 1]
                dst[y*dst_w + x] = w1*p1 + w2*p2 + w3*p3 + w4*p4
            }
        }
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        //print("Upsample", inputs[0].shape, outputs[0].shape)
        
        for i in 0..<inputs[0].shape[2].intValue {
            bilinear(src_mlma: inputs[0], dst_mlma: outputs[0], index: i, src_w: inputs[0].shape[4].intValue, src_h: inputs[0].shape[3].intValue, dst_w: outputs[0].shape[4].intValue, dst_h: outputs[0].shape[3].intValue)
        }
    }
}
