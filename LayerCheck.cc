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

static void set_inputs(Layer* l0){
	l0->output[0] = 0.05; l0->output[1] = 0.10;
}

int main(){

	dimen siz0 = {1, {2, 0, 0, 0}};
	dimen siz1 = {1, {2, 0, 0, 0}};
	dimen siz2 = {1, {2, 0, 0, 0}};
	Layer *l0 = new InputLayer(     siz0);
	Layer *l1 = new FullLayer(siz0, siz1);
	Layer *l2 = new FullLayer(siz1, siz2);

	l1->setActivation(act);
	l1->setDerActivation(deract);
	l2->setActivation(act);
	l2->setDerActivation(deract);

	init_weights(l1, l2);
	set_inputs(l0);

//	l0->forward();
	l1->forward(l0);
	l2->forward(l1);

	l2->DLayer[0] = l2->output[0] - 0.01;
	l2->DLayer[1] = l2->output[1] - 0.99;

	l2->backward(l1);
	l1->backward(l0);

	std::cout << l2->DWeight[0] <<std::endl;
	std::cout << l2->DWeight[1] <<std::endl;
	std::cout << l2->DWeight[2] <<std::endl;
	std::cout << l2->DWeight[3] <<std::endl;


	std::cout << "shithead"<<std::endl;
	return 0;
}

