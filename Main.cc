#include <iostream>
#include <math.h>
#include "MLP.h"

double act(double x){
	double y = exp(-x);
	return 1.0/(1.0+y);
}

double deract(double x){
	return x*(1-x);
}

static void init_weights(Layer *l1, Layer *l2){

	l1->weights[0] = 0.15; l1->weights[1] = 0.25;
	l1->weights[2] = 0.20; l1->weights[3] = 0.30;
	l1->weightsb[0] = 0.35; l1->weightsb[1] = 0.35;

	l2->weights[0] = 0.40; l2->weights[1] = 0.50;
	l2->weights[2] = 0.45; l2->weights[3] = 0.55;
	l2->weightsb[0] = 0.60; l2->weightsb[1] = 0.60;
}


int main(){

	dimen siz0 = {1, {2, 0, 0, 0}};
	Layer *l0 = new InputLayer(     siz0);

	dimen siz1 = {1, {2, 0, 0, 0}};
	Layer *l1 = new FullLayer(siz0, siz1);
	l1->setActivation(act);
	l1->setDerActivation(deract);

	dimen siz2 = {1, {2, 0, 0, 0}};
	Layer *l2 = new FullLayer(siz1, siz2);	
	l2->setActivation(act);
	l2->setDerActivation(deract);

	init_weights(l1, l2);

	Network net;
	net.addLayer(l0);
	net.addLayer(l1);
	net.addLayer(l2);

	double inputs[2] = {0.05, 0.10};
	double goldenOutput[2] = {0.01, 0.99};
	net.assignInput(inputs);
	net.forwprop();
	net.derOutputSquaredError(goldenOutput);
	net.backprop();

	std::cout << l2->DWeight[0] <<std::endl;
	std::cout << l2->DWeight[1] <<std::endl;
	std::cout << l2->DWeight[2] <<std::endl;
	std::cout << l2->DWeight[3] <<std::endl;


	std::cout << "shithead"<<std::endl;
	return 0;
}

