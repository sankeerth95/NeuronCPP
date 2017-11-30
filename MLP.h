#ifndef MLP
#define MLP
#include <vector>

typedef struct dims{
   int dim;
   int lens[4];
   void operator=(const struct dims &prev){
	dim = prev.dim;
	lens[0] = prev.lens[0];
	lens[1] = prev.lens[1];
	lens[2] = prev.lens[2];
	lens[3] = prev.lens[3];
   }
}dimen;

class Layer{
public:
   double *output, *DLayer, *weights, *DWeight, *weightsb, *DWeightb;   

   int weightssize, weightsbsize;
   double *increments;

   dimen prevdim, dim;
   double (*activation)(double x);
   double (*derAct)(double x);
public:
   Layer(dimen prev, dimen current);
   ~Layer();
   void setActivation(double (*activ_pointer)(double) );
   void setDerActivation(double (*activ_pointer)(double) );
   void zeroIncrement();
   virtual void forward(const Layer *prev)=0;
   virtual void backward(const Layer *prev)=0;
   virtual void updateIncrement()=0;
   virtual void updateWeights(int batchsize)=0;

};

class InputLayer : public Layer{
public:
   InputLayer(dimen current);
   ~InputLayer();
   void forward(const Layer *prev);
   void backward(const Layer *prev);
   void updateIncrement();
   void updateWeights(int batchsize);
};

class FullLayer : public Layer{
public:
   FullLayer(dimen prevDim, dimen current);
   ~FullLayer();
   void	forward(const Layer *prev);
   void backward(const Layer *next);
   void updateIncrement();
   void updateWeights(int batchsize);
};

class Network{
private:
   int numLayers;
   std::vector<Layer*> layers;
public:
   Network();
   ~Network();
   void assignInput(double* img);
   void addLayer(Layer *l);

   void forwprop();
   void derOutputSquaredError(double *);
   void backprop();

   double* getOutput();

   void step(int batchsize);

   void zeroIncrements();
   void initWeights();

   void updateIncrements();
   void updateWeights(int batchsize);
//   void train();
};

#endif

