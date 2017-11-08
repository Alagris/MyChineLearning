//
//  training_batch.hpp
//  MyChineLearning
//
//  Created by Alagris on 01/11/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef training_batch_hpp
#define training_batch_hpp
#include <basic_types.h>
#include <fixed_size_dynamic_array.hpp>
namespace mchln {
    class training_batch{
    public:
        training_batch(const size_t expectedSize,const size_t inputSize,const size_t series):
        m_expectedSize(expectedSize),
        m_inputSize(inputSize),
        m_series(series),
        m_expected(m_expectedSize*series),
        m_inputs(m_inputSize*series){}
        
        const mymath::precise expectedAt(const size_t seriesIndex,const size_t index)const{
            return const_cast<training_batch*>(this)->expectedAt(seriesIndex,index);
        }
        const mymath::precise * const expectedSeriesAt(const size_t seriesIndex)const{
            return const_cast<training_batch*>(this)->expectedSeriesAt(seriesIndex);
        }
        const mymath::precise inputAt(const size_t seriesIndex,const size_t inputIndex)const{
            return const_cast<training_batch*>(this)->inputAt(seriesIndex, inputIndex);
        }
        const mymath::precise * const inputSeriesAt(const size_t seriesIndex)const{
            return const_cast<training_batch*>(this)->inputSeriesAt(seriesIndex);
        }
        mymath::precise& expectedAt(const size_t seriesIndex,const size_t expectedIndex){
            return m_expected.get()[seriesIndex*m_inputSize+expectedIndex];
        }
        mymath::precise * const expectedSeriesAt(const size_t seriesIndex){
            return m_expected.get()+seriesIndex*m_expectedSize;
        }
        mymath::precise& inputAt(const size_t seriesIndex,const size_t inputIndex){
            return m_inputs.get()[seriesIndex*m_expectedSize+inputIndex];
        }
        mymath::precise * const inputSeriesAt(const size_t seriesIndex){
            return m_inputs.get()+seriesIndex*m_inputSize;
        }
    private:
        size_t m_expectedSize;
        size_t m_inputSize;
        size_t m_series;
        util_lib::fixed_size_dynamic_array<mymath::precise> m_expected;
        util_lib::fixed_size_dynamic_array<mymath::precise> m_inputs;
    };
}
#endif /* training_batch_hpp */
