//
//  layered_network.hpp
//  MyChineLearning
//
//  Created by Alagris on 28/10/2017.
//  Copyright © 2017 alagris. All rights reserved.
//

#ifndef layered_network_hpp
#define layered_network_hpp
#include <mat.hpp>
#include "../math/activation_function.h"
#include <shared_ptr_empty_destructor.h>
namespace mchln {
    class layer{
    public:
        layer(const size_t neuronsNumber,
              const size_t neuronsNumberInNextLayer,
              mymath::precise * const neuronBuffer=nullptr,
              mymath::precise * const weightBuffer=nullptr,
              const std::shared_ptr<layer>& prev=nullptr):
        m_neurons(neuronsNumber+(neuronsNumberInNextLayer>0),1,neuronBuffer),
        m_weights(neuronsNumberInNextLayer,m_neurons.size(),weightBuffer),
        m_prev(prev){}
        
        friend  std::shared_ptr<layer> addLayer(std::shared_ptr<layer> & thisLayer,
                                                const size_t neuronsNumberInLayerAfterNext,
                                                mymath::precise * const neuronBuffer,
                                                mymath::precise * const weightBuffer);
        
        void assign(const mymath::mat & neurons,const mymath::mat & weights){
            m_neurons.assign(neurons,weights);
        }
        const bool containsBias()const{
            return m_weights.columns()>0;
        }
        const size_t getNeuronsInNextlayer()const{
            return m_weights.columns();
        }
        const size_t size()const{
            return m_neurons.size();
        }
        const size_t sizeWithoutBias()const{
            return size()-containsBias();
        }
        const size_t biasIndex()const{
            return m_neurons.size()-containsBias();
        }
        const mymath::precise bias()const{
            return m_neurons.at(biasIndex());
        }
        mymath::precise& bias(){
            return m_neurons.at(biasIndex());
        }
        const std::shared_ptr<const layer> next()const{
            return m_next;
        }
        std::shared_ptr<layer> next(){
            return m_next;
        }
        const bool hasNext()const{
            return m_next.get()!=nullptr;
        }
        const std::weak_ptr<const layer> prev()const{
            return m_prev;
        }
        std::weak_ptr<layer> prev(){
            return m_prev;
        }
        const bool hasPrev()const{
            return !m_prev.expired();
        }
        mymath::precise operator[](const size_t i)const{
            return neuron(i);
        }
        mymath::precise& operator[](const size_t i){
            return neuron(i);
        }
        /**neurons store number that corresponds to input sum (not the activation function output!)*/
        const mymath::precise neuron(const size_t index)const{
            return neurons().at(index);
        }
        /**neurons store number that corresponds to input sum (not the activation function output!)*/
        mymath::precise& neuron(const size_t index){
            return m_neurons.at(index);
        }
        const mymath::precise weight(const size_t column,const size_t row)const{
            return m_weights.get(column,row);
        }
        mymath::precise& weight(const size_t column,const size_t row){
            return m_weights.get(column,row);
        }
        const mymath::precise weightTo(const size_t thisNeuronIndex,const size_t nextNeuronIndex)const{
            return weight(nextNeuronIndex, thisNeuronIndex);
        }
        mymath::precise& weightTo(const size_t thisNeuronIndex,const size_t nextNeuronIndex){
            return weight(nextNeuronIndex, thisNeuronIndex);
        }
        const mymath::precise biasTo(const size_t nextNeuronIndex)const{
            return weightTo(biasIndex(), nextNeuronIndex);
        }
        mymath::precise& biasTo(const size_t nextNeuronIndex){
            return weightTo(biasIndex(),nextNeuronIndex);
        }
        const size_t weightsPerNeuron()const{
            return weights().columns();
        }
        template<typename T>
        layer & operator=(T*const array){
            m_neurons=array;
            return *this;
        }
        const mymath::mat & neurons()const{
            return m_neurons;
        }
        const mymath::mat & weights()const{
            return m_weights;
        }
        const mymath::precise *const neuronsPtr()const{
            return m_neurons.ptr();
        }
        const mymath::precise *const weightsPtr()const{
            return m_weights.ptr();
        }
        mymath::precise *const neuronsPtr(){
            return m_neurons.ptr();
        }
        mymath::precise *const weightsPtr(){
            return m_weights.ptr();
        }
    private:
        const bool setNext(const size_t neuronsNumber,
                           const size_t neuronsNumberInNextLayer,
                           mymath::precise * const neuronBuffer=nullptr,
                           mymath::precise * const weightBuffer=nullptr,
                           const std::shared_ptr<layer> & prev=nullptr){
            if(m_next)return false;
            m_next.reset(new layer(neuronsNumber,neuronsNumberInNextLayer,neuronBuffer,weightBuffer,prev));
            return true;
        }
        mymath::mat m_neurons;
        mymath::mat m_weights;
        std::shared_ptr<layer> m_next;
        std::weak_ptr<layer> m_prev;
    };
    
    
    
    inline std::shared_ptr<layer> addLayer(std::shared_ptr<layer> & thisLayer,
                                           const size_t neuronsNumberInLayerAfterNext,
                                           mymath::precise * const neuronBuffer=nullptr,
                                           mymath::precise * const weightBuffer=nullptr){
        if(thisLayer->setNext(thisLayer->getNeuronsInNextlayer(),neuronsNumberInLayerAfterNext,neuronBuffer,weightBuffer,thisLayer)){
            return thisLayer->next();
        }else{
            return nullptr;
        }
    }
    
}
#endif /* layered_network_hpp */
