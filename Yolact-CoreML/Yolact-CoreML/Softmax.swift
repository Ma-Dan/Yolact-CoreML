//
//  Softmax.swift
//  Yolact-CoreML
//
//  Created by 马丹 on 2019/6/15.
//  Copyright © 2019 MachineThink. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(Softmax) class Softmax: NSObject, MLCustomLayer {
    
    required init(parameters: [String : Any]) throws {
        print(#function, parameters)
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //print(#function, weights)
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        print("Softmax output shape", inputShapes)
        
        return inputShapes
    }
    
    func softmax(x: UnsafeMutablePointer<Float>, len: UInt) {
        // Find the maximum value in the input array.
        var max: Float = 0
        vDSP_maxv(x, 1, &max, len)
        
        // Subtract the maximum from all the elements in the array.
        // Now the highest value in the array is 0.
        max = -max
        vDSP_vsadd(x, 1, &max, x, 1, len)
        
        // Exponentiate all the elements in the array.
        var count = Int32(len)
        vvexpf(x, x, &count)
        
        // Compute the sum of all exponentiated values.
        var sum: Float = 0
        vDSP_sve(x, 1, &sum, len)
        
        // Divide each element by the sum. This normalizes the array contents
        // so that they all add up to 1.
        vDSP_vsdiv(x, 1, &sum, x, 1, len)
    }
    
    func calcSoftmax(inputs: [MLMultiArray], outputs: [MLMultiArray]) {
        let src = UnsafeMutablePointer<Float>(OpaquePointer(inputs[0].dataPointer))
        var dst = UnsafeMutablePointer<Float>(OpaquePointer(outputs[0].dataPointer))
        
        for i in 0..<inputs[0].count {
            dst[i] = src[i]
        }
        
        let count = inputs[0].shape[2].intValue
        let len = inputs[0].shape[3].intValue
        for _ in 0..<count {
            softmax(x: dst, len: (UInt)(len))
            dst = dst.advanced(by: len)
        }
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        //print("Softmax", inputs[0].shape, outputs[0].shape)
        calcSoftmax(inputs: inputs, outputs: outputs)
    }
}
