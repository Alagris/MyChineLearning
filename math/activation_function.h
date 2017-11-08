//
//  activation_function.h
//  MyChineLearning
//
//  Created by Alagris on 31/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef activation_function_h
#define activation_function_h

#include <function.hpp>
namespace mchln{
    namespace{
        inline constexpr const mymath::precise identity_f(const  mymath::precise s){
            return s;
        }
        inline constexpr const mymath::precise identity_d(const  mymath::precise s){
            return 1;
        }
        inline const  mymath::precise sigmoid_f(const  mymath::precise s){
            return 1/(1+exp(-s));
        }
        inline const  mymath::precise sigmoid_d(const  mymath::precise s){
            return sigmoid_f(s)*(1-sigmoid_f(s));
        }
    }
    template<constexpr const mymath::one_arg_func f,constexpr const mymath::one_arg_func d>
    struct activation_function{
        /**Function f(s)=y*/
        static constexpr const mymath::one_arg_func func=f;
        /**Derivative dy/ds*/
        static constexpr const mymath::one_arg_func der=d;
    };
    
    typedef activation_function<identity_f,identity_d>identity_activation_func;
    typedef activation_function<sigmoid_f,sigmoid_d>sigmoid_activation_func;
}

#endif /* activation_function_h */
