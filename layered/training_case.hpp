//
//  training_case.hpp
//  MyChineLearning
//
//  Created by Alagris on 01/11/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef training_case_hpp
#define training_case_hpp
#include "layered_network.hpp"
#include "layered_network_helpers.hpp"
#include "error_function.h"
#include <fixed_size_dynamic_array.hpp>
#include "training_batch.hpp"
namespace mchln {
    class training_batch_keeper{
        
    };
    
    
    class training_case_keeper{
    public:
        training_case_keeper(const std::shared_ptr<layer> & input):
        m_input(input),
        m_output(getLastLayer(input)),
        m_error(m_output->size()){}
        template<typename error_func,typename activation_func>
        void calculateError(const training_batch & batch,const size_t seriesIndex){
            calculateError<error_func,activation_func>(batch.inputSeriesAt(seriesIndex),batch.expectedSeriesAt(seriesIndex));
        }
        template<typename error_func,typename activation_func>
        void calculateError(const mymath::precise*const input,const mymath::precise*const expectedOutput){
            layer & in=inputRef();
            in=input;
            const layer & out=propagateToLastLayer<activation_func>(in);
            for(size_t i=0;i<out.size();i++){
                getError(i)+=error_func::func(out[i],expectedOutput[i]);
            }
        }
        const mymath::precise getError(const size_t index)const{
            return m_error.get()[index];
        }
        mymath::precise & getError(const size_t index){
            return m_error.get()[index];
        }
        const size_t getErrorSize()const{
            return m_output->size();
        }
        void clearError(){
            for(size_t i=0;i<getErrorSize();i++){
                m_error.get()[i]=0;
            }
        }
        const std::shared_ptr<const layer> input()const{
            return m_input;
        }
        const std::shared_ptr<const layer> output()const{
            return m_output;
        }
        const layer & inputRef()const{
            return *m_input.get();
        }
        const layer & outputRef()const{
            return *m_output.get();
        }
        
    private:
        layer & inputRef(){
            return *m_input.get();
        }
        layer & outputRef(){
            return *m_output.get();
        }
        std::shared_ptr<layer> m_input;
        std::shared_ptr<layer> m_output;
        util_lib::fixed_size_dynamic_array<mymath::precise> m_error;
    };
    
    
}
#endif /* training_case_hpp */
