//
//  layered_network_helpers.hpp
//  MyChineLearning
//
//  Created by Alagris on 29/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef layered_network_helpers_hpp
#define layered_network_helpers_hpp
#include "layered_network.hpp"
#include "../math/error_function.h"
//#include <iostream>
namespace mchln{
    void randomizeWeights(layer&l,const mymath::precise min,const mymath::precise max);
    void randomizeNeurons(layer&l,const mymath::precise min,const mymath::precise max);
    inline const std::shared_ptr<layer> getLastLayer(std::shared_ptr<layer> l){
        while(true){
            if(l->next()){
                l=l->next();
            }else{
                return l;
            }
        }
    }
    
    void printLayer(const layer&l);
    void printAllLayers(const layer&l);
    void setAllBiases(layer&l,const mymath::precise bias);

    template<typename activation_func>
    inline mymath::mat computeNeuronsOutput(const layer&l){
        mymath::mat output(l.size(),1);
        size_t i=0;
        for(;i<l.sizeWithoutBias();i++){
            output.at(i)=activation_func::func(l.neuron(i));
//            std::cout<<l.neuron(i)<<" "<<output.at(i)<<"\n";
        }
        if(l.containsBias()){
            //bias is at the end
            output.at(i)=l.neuron(i);
        }
        return output;
    }
    template<typename activation_func>
    inline std::shared_ptr<layer> propagateToNextLayer(layer&l){
        if(l.hasNext()){
            l.next()->assign(computeNeuronsOutput<activation_func>(l),l.weights());
            return l.next();
        }
        return nullptr;
    }
    template<typename activation_func>
    inline std::shared_ptr<layer> propagateToLastLayer(layer&l){
        std::shared_ptr<layer> current = propagateToNextLayer<activation_func>(l);
        while(true){
            std::shared_ptr<layer> next = propagateToNextLayer<activation_func>(*current);
            if(next){
                current=next;
            }else{
                return current;
            }
        }
    }
    
    inline const mymath::precise derivativeOf_InputSum_WithRespectTo_PreviousNeuronOutput(const layer * previousLayer,
                                                                                   const size_t previousNeuronIndex,
                                                                                   const size_t thisNeuronIndex){
        //sum:
        // s=w0*x0+w1*x1+...
        //derivative:
        // ds/dxi=
        // d/dxi(w0*x0+w1*x1+...)=
        // d/dxi(wi*xi)=
        // wi
        return previousLayer->weightTo(previousNeuronIndex, thisNeuronIndex);
    }
    template<typename activ_func>
    inline const mymath::precise derivativeOf_NeuronOutput_WithRespectTo_ThisNeuronInputSum(const mymath::precise inputSum){
        //sum:
        // s=w0*x0+w1*x1+...
        //output:
        // y=f(s)
        //f is activation function
        //derivative:
        // dy/ds=d/ds(f(s))=f'(s)
        return activ_func::der(inputSum);
    }
    template<typename activ_func>
    inline const mymath::precise derivativeOf_NeuronOutput_WithRespectTo_ThisNeuronInputSum(const layer * thisLayer,
                                                                                     const size_t thisNeuronIndex){
        return derivativeOf_NeuronOutput_WithRespectTo_ThisNeuronInputSum<activ_func>(thisLayer->neuron(thisNeuronIndex));
    }
}
#endif /* layered_network_helpers_hpp */
