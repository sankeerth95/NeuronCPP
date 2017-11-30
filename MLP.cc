#include <iostream>
#include <vector>
#include "MLP.h"

static double defaultAct(double x){
   return x;
}
static int getSize(const dimen& d){
   int size = 1;
   for(int i = 0; i < d.dim; i++)
	size *= d.lens[i];
   return size;
}

//Layer constructor
Layer::Layer(const dimen prev, const dimen current){
   prevdim = prev;
   dim = current;
   activation=defaultAct;
   output = new double[getSize(current)];
   DLayer = new double[getSize(current)];
}

Layer::~Layer(){
   delete output;
   delete DLayer;
}

void Layer::setActivation(double (*activ_pointer)(double) ){
   activation = activ_pointer;
}

void Layer::setDerActivation(double (*activ_pointer)(double) ){
   derAct = activ_pointer;
}

void Layer::zeroIncrement(){
   for(int i = 0; i < weightssize+weightsbsize; i++)
      increments[i] = 0.0;
}


//input layer
static dimen d0 = {0, {0, 0, 0, 0}};
InputLayer::InputLayer(dimen current) : Layer(d0, current){
   weightssize = 0;
   weightsbsize = 0;
}
InputLayer::~InputLayer(){}
void InputLayer::forward(const Layer *prev){}
void InputLayer::backward(const Layer *prev){}
void InputLayer::updateIncrement(){}
void InputLayer::updateWeights(int batchsize){}


//Fully Connected Layer
FullLayer::FullLayer(dimen prev, dimen current) : Layer(prev, current){

   weightssize = getSize(current)*getSize(prev) ;
   weightsbsize = getSize(current);

   increments = new double[getSize(current)*getSize(prev) + getSize(current)];
   weights = new double[getSize(current)*getSize(prev)];
   weightsb = new double[getSize(current)];
   DWeight = new double[getSize(current)*getSize(prev)];
   DWeightb = new double[getSize(current)];
}

FullLayer::~FullLayer(){
   delete increments;
   delete weights;
   delete weightsb;
   delete DWeight;
   delete DWeightb;
}

void FullLayer::forward(const Layer *prev){

   double *out = output;
   for(int j = 0; j < getSize(this->dim); j++){
      *out = 0.0;
      for(int i = 0; i < getSize(prev->dim); i++){
	 *out += prev->output[i]*weights[i*getSize(this->dim)+j];  
      }
      *out += weightsb[j];
      *out = activation(*out);
      out++;
   }
}

void FullLayer::backward(const Layer *prev){

   double *predlayer = prev->DLayer;
   for(int i = 0; i < getSize(prev->dim); i++){
      *predlayer = 0.0;
      for(int j = 0; j < getSize(this->dim); j++){
         *predlayer += DLayer[j]*(weights[i*getSize(this->dim)+j])*derAct(output[j]);
         DWeight[i*getSize(this->dim)+j] = derAct(output[j])*DLayer[j]*(prev->output[i]);
      }
      predlayer++;
   }
   for(int j = 0; j < getSize(dim); j++){
      DWeightb[j] = derAct(output[j])*DLayer[j];
   }
}

void FullLayer::updateIncrement(){
   for(int i = 0; i < getSize(prevdim); i++){
      for(int j = 0; j < getSize(this->dim); j++){
         increments[i*getSize(this->dim)+j] += DWeight[i*getSize(this->dim)+j];
      }
   }
   for(int j = 0; j < getSize(dim); j++){
      increments[getSize(prevdim)*getSize(this->dim) + j] += DWeightb[j];
   }
}

void FullLayer::updateWeights(int batchsize){
   for(int i = 0; i < getSize(dim)*getSize(prevdim); i++){
      weights[i] -= increments[i]/((double)batchsize);
   }
   for(int i = 0; i < getSize(dim); i++){
      weightsb[i] -= increments[getSize(dim)*getSize(prevdim) + i]/((double)batchsize);
   }
}

//Network
Network::Network(){
   numLayers = 0;
}

Network::~Network(){
//   g3t r3kt u scrUb;
}

void Network::assignInput(double *img){
       layers[0]->output = img;
}

void Network::addLayer(Layer *l){
   layers.push_back(l);
   numLayers++;
}

void Network::forwprop(){
   for(int i = 1; i < numLayers; i++){
      layers[i]->forward(layers[i-1]);
   }
}

void Network::derOutputSquaredError(double *gold){
   for(int i = 0; i < getSize(layers[numLayers-1]->dim); i++){
      layers[numLayers-1]->DLayer[i] = layers[numLayers-1]->output[i] - gold[i];
   }
}

void Network::backprop(){
   for(int i = numLayers-1; i >= 1; i--){
      layers[i]->backward(layers[i-1]);
   }
}

double* Network::getOutput(){
   return layers[numLayers-1]->output;
}

void Network::initWeights(){
   for(int i = 1; i < numLayers; i++){
   //   layers[i]->initWeights();
   }
}

void Network::zeroIncrements(){
   for(int i = 1; i < numLayers; i++)
      layers[i]->zeroIncrement();
}

void Network::step(int batchsize){	//incomplete: doesnt take new input
   zeroIncrements();
   for(int i = 0; i < batchsize; i++){
//      assignInput();
      forwprop();
 //     derOutputSquaredError();
      backprop();
      updateIncrements();
   }
   updateWeights(batchsize);
}

void Network::updateIncrements(){
   for(int i = 1; i < numLayers; i++){
      layers[i]->updateIncrement();
   }
}

void Network::updateWeights(int batchsize){
   for(int i = 1; i < numLayers; i++){
      layers[i]->updateWeights(batchsize);
   }
}


