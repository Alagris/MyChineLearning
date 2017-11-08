//
//  error_function.h
//  MyChineLearning
//
//  Created by Alagris on 31/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef error_function_h
#define error_function_h

#include <function.hpp>
#include <math.h>
namespace mchln{
    
    namespace{
        inline const mymath::precise square_error_f(const mymath::precise output,const mymath::precise expected){
            const mymath::precise diff=expected-output;
            return diff*diff;
        }
        inline const mymath::precise square_error_d(const mymath::precise output,const mymath::precise expected){
            return -2*(expected-output)*output*(1-output);
        }
        
        /**expected should be either 1 or 0*/
        inline const mymath::precise log_error_f(const mymath::precise output,const mymath::precise expected){
            if(expected==0)return -logl(1-output);
            //else expected should be 1
            return -logl(output);
        }
        /**expected should be either 1 or 0*/
        inline const mymath::precise log_error_d(const mymath::precise output,const mymath::precise expected){
            if(expected==0)return 1/output;
            //else expected should be 1
            return -1/output;
        }

    }
    
    /**Computes: output --->  error
     (for single training case and single output neuron)*/
    template<constexpr const mymath::two_arg_func f,constexpr const mymath::two_arg_func d>
    struct error_function{
        static constexpr const mymath::two_arg_func func=f;
        static constexpr const mymath::two_arg_func der=d;
    };
    
    /**Computes: array of output neuron errors ---> combined error
     (for single training case)*/
    template<constexpr const mymath::arr_arg_func case_f,constexpr const mymath::arr_arg_func case_d>
    struct training_case_error_function{
        static constexpr const mymath::arr_arg_func func=case_f;
        static constexpr const mymath::arr_arg_func der=case_d;
    };
    
    /**Computes: array of combined errors ---> batch error*/
    template<constexpr const mymath::arr_arg_func batch_f,constexpr const mymath::arr_arg_func batch_d>
    using batch_error_function=training_case_error_function<batch_f,batch_d>;
    
    typedef error_function<square_error_f,square_error_d> square_error;
    typedef error_function<log_error_f,log_error_d> log_error;
    
    

}
#endif /* error_function_h */
