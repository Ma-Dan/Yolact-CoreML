//
//  Concat.swift
//  Yolact-CoreML
//

import Foundation
import CoreML

@objc(Concat) class Concat: NSObject, MLCustomLayer {

    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //print(#function, weights)
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        print("Concat input shape", inputShapes)
        
        if inputShapes[0][0].intValue == 0 {
            return inputShapes
        }
        
        var height = 0
        
        for i in 0..<5 {
            height = height + inputShapes[i][3].intValue
        }
        let width = inputShapes[0][4].intValue
        
        //Workaround for MTLTextureDescriptor width height max allowed size 16384
        let outputShape = [[1, 1, NSNumber(value: height), NSNumber(value: width), 1]]
        
        print("Concat output shape", outputShape)
    
        return outputShape
    }
    
    func concat(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
        var dst = UnsafeMutablePointer<Float>(OpaquePointer(outputs[0].dataPointer))
        
        for i in 0..<inputs.count {
            let src = UnsafeMutablePointer<Float>(OpaquePointer(inputs[i].dataPointer))
            let length = inputs[i].count
            for j in 0..<length {
                dst[j] = src[j]
            }
            dst = dst.advanced(by: length)
        }
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        //print("Concat", inputs.count, outputs.count)
        //print("Concat", inputs[0].count, inputs[1].count, inputs[2].count, inputs[3].count, inputs[4].count)
        concat(inputs: inputs, outputs: outputs)
    }
}
