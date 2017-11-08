//
//  backpropagation_helpers.hpp
//  MyChineLearning
//
//  Created by Alagris on 29/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef backpropagation_helpers_hpp
#define backpropagation_helpers_hpp
#include "backpropagation.hpp"
#include "training_batch.hpp"
#include <vector_pair.hpp>
namespace mchln {
    
    
    
    
    

//    template<typename err_func,typename activ_func>
//    void backpropagate(backpropagation & bp,const training_batch &b,const size_t biggestLayerSize=0){
//        bp.batchUpdateError<err_func::func, activ_func::func>(b);
        
        
//        const layer * l = bp.output().get();
//        
//        util_lib::vector_pair<mymath::precise> cache(biggestLayerSize?biggestLayerSize:bp.input()->size(),0);
//        
//        
//        while(true){
//            const layer * prev=l->prev().get();
//            for(size_t i=0;i<prev->size();i++){
//                
//            }
//            l=prev;
//            cache.swap();
//        }
//    }
}
#endif /* backpropagation_helpers_hpp */
