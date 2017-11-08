//
//  regular_network.hpp
//  MyChineLearning
//
//  Created by Alagris on 29/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//
#ifndef regular_network_hpp
#define regular_network_hpp
#include "layered_network.hpp"
#include <vector>


namespace mchln{
    class regnet{
    public:
        /**networkDepth - number of hidden layers + input layer but excluding output layer*/
        regnet(const size_t layerWidth,const size_t networkDepth,const size_t outputWidth):
        m_width(layerWidth),
        m_depth(networkDepth),
        m_outputSize(outputWidth),
        m_buffer(new mymath::precise[bufferSize()]),
        m_inputLayer(new layer(width(0),width(1),getNeuron(0),getWeight(0))),
        m_outputLayer(m_inputLayer){
            //            m_net.reserve(depth()+1);
            //            initFirstLayer();
            //            for(size_t layerIndex=1;layerIndex<depth()-1;layerIndex++){
            //                initNextLayer();
            //            }
            //            initNextLayer(outputWidth);
            //            initNextLayer(0);
            ;
            for(size_t layerIndex=1;layerIndex<=depth();layerIndex++){
                initNextLayer(layerIndex);
            }
            
        }
        
    private:
         void initNextLayer(const size_t layerIndex){
            m_outputLayer=addLayer(m_outputLayer,width(layerIndex+1),getNeuron(layerIndex),getWeight(layerIndex));
        }
    public:
        //         layer & operator[](const size_t i){
        //            return *(m_net[i]);
        //        }
        //         const layer & operator[](const size_t i)const {
        //            return *(m_net[i]);
        //        }
        //         layer * at(const size_t i){
        //            return m_net[i].get();
        //        }
        //         const layer * at(const size_t i)const {
        //            return m_net[i].get();
        //        }
        //         const layer * output()const {
        //            return m_net[depth()].get();
        //        }
        const std::shared_ptr<const layer> input()const{
            return m_inputLayer;
        }
        const std::shared_ptr<const layer> output()const{
            return m_outputLayer;
        }
        layer *const input(){
            return m_inputLayer.get();
        }
        layer *const output(){
            return m_outputLayer.get();
        }
         const size_t depth()const{
            return m_depth;
        }
         const size_t hiddenAndInputLayers()const{
            return depth();
        }
         const size_t totalLayersNumber()const{
            return hiddenAndInputLayers()+1;
        }
         const size_t width()const{
            return m_width;
        }
         const size_t width(const size_t layerIndex)const{
            if(layerIndex==depth())return outputNeuronsNumber();
            if(layerIndex>depth())return 0;
            return width();
        }
         const size_t neuronsPerLayer()const{
            return width();
        }
         const size_t neuronsPerLayerPlusBias()const{
            return neuronsPerLayer()+1;
        }
         const size_t weightsPerNeuron()const{
            return neuronsPerLayerPlusBias();
        }
         const size_t weightsPerLayer()const{
            return neuronsPerLayer()*weightsPerNeuron();
        }
        /**returns number of hidden and input neurons*/
         const size_t neuronsNumber()const{
            return neuronsPerLayer()*hiddenAndInputLayers();
        }
         const size_t neuronsNumberIncludingBias()const{
            return neuronsPerLayerPlusBias()*hiddenAndInputLayers();
        }
         const size_t outputNeuronsNumber()const{
            return m_outputSize;
        }
         const size_t totalNeuronsNumber()const{
            return neuronsNumberIncludingBias()+outputNeuronsNumber();
        }
         const size_t outputWeightsNumber()const{
            return outputNeuronsNumber()*weightsPerNeuron();
        }
         const size_t weightsNumber()const{
            return weightsPerLayer()*hiddenAndInputLayers();
        }
         const size_t totalWeightsNumber()const{
            return weightsNumber()+outputWeightsNumber();
        }
         const size_t bufferSize()const{
            return totalNeuronsNumber()+totalWeightsNumber();
        }
         const mymath::precise * const getNeuron(const size_t layer,const size_t index=0)const{
            return const_cast<regnet *>(this)->getNeuron(layer,index);
        }
         const mymath::precise * const getOutputNeuron(const size_t index=0)const{
            return const_cast<regnet *>(this)->getOutputNeuron(index);
        }
         const mymath::precise * const getWeight(const size_t layer,const size_t neuronIndex,const size_t weightIndex=0)const{
            return const_cast<regnet *>(this)->getWeight(layer,neuronIndex,weightIndex);
        }
         const mymath::precise * const getOutputNeuronWeight(const size_t neuronIndex=0,const size_t weightIndex=0)const{
            return const_cast<regnet *>(this)->getOutputNeuronWeight(neuronIndex,weightIndex);
        }
    private:
         mymath::precise * const getNeuron(const size_t layer,const size_t index=0){
            return m_buffer.get()+layer*neuronsPerLayerPlusBias()+index;
        }
         mymath::precise * const getOutputNeuron(const size_t index=0){
            return getNeuron(hiddenAndInputLayers(),index);
        }
         mymath::precise * const getNeuronsEndAndWeightsBeginning(){
            return getOutputNeuron(outputNeuronsNumber());
        }
         mymath::precise * const getWeight(const size_t layer,const size_t neuronIndex=0,const size_t weightIndex=0){
            return getNeuronsEndAndWeightsBeginning()+layer*weightsPerLayer()+neuronIndex*weightsPerNeuron()+weightIndex;
        }
         mymath::precise * const getOutputNeuronWeight(const size_t neuronIndex=0,const size_t weightIndex=0){
            return getWeight(hiddenAndInputLayers(),neuronIndex,weightIndex);
        }
        //         void initNextLayer(){
        //            initNextLayer(width());
        //        }
        //         void initNextLayer(const size_t nextLayerWidth){
        //            const size_t last=m_net.size();
        //            m_net.push_back(at(last-1)->addLayer(nextLayerWidth,getNeuron(last),getWeight(last)));
        //        }
        //         void initFirstLayer(){
        //            m_net.push_back(std::shared_ptr<layer>(new layer(width(),width(),getNeuron(0),getWeight(0))));
        //        }
        
        
        //        std::vector<std::shared_ptr<layer>> m_net;
        size_t m_width;
        size_t m_depth;
        size_t m_outputSize;
        /*Layout:
         -depth times
         -  width times - layer
         -      neuron
         -output_width times
         -  output neuron
         -depth-1 times
         -  width-1 times - layer
         -      width times - neuron
         -          weight
         -output_width times
         -  output weight
         */
        std::shared_ptr<mymath::precise> m_buffer;
        std::shared_ptr<layer> m_inputLayer;
        std::shared_ptr<layer> m_outputLayer;
    };
}
#endif /* regular_network_hpp */
