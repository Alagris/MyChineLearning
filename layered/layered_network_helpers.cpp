//
//  layered_network_helpers.cpp
//  MyChineLearning
//
//  Created by Alagris on 29/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#include "layered_network_helpers.hpp"
#include "random.hpp"
#include <iostream>
namespace mchln{
    void randomizeWeights(layer&l,const mymath::precise min,const mymath::precise max){
        layer * current=&l;
        while(current){
            mymath::generateRandArray(current->weightsPtr(), current->weights().size(), min, max);
            current=current->next().get();
        }
    }
    void randomizeNeurons(layer&l,const mymath::precise min,const mymath::precise max){
        layer * current=&l;
        while(current){
            mymath::generateRandArray(current->neuronsPtr(), current->neurons().size(), min, max);
            current=current->next().get();
        }
    }
    
    void printLayer(const layer&l){
        if(l.size()==0)return;
        std::cout<<l[0];
        for(size_t i=1;i<l.size();i++){
            std::cout<<"|"<<l[i];
        }
        std::cout<<"\n";
    }
    void printAllLayers(const layer&l){
        const layer * current=&l;
        while(current){
            printLayer(*current);
            current=current->next().get();
        }
    }
    void setAllBiases(layer&l,const mymath::precise bias){
        layer * current=&l;
        while(current){
            if(current->containsBias())current->bias()=bias;
            current=current->next().get();
        }
    }
}
