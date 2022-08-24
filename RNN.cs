using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;
using static NeatNetwork.Libraries.Activation;

namespace NeatNetwork
{
    public class RNN
    {
        public int Length => Neurons.Count;
        public int InputLength => Neurons[0][0].Connections.Length;
        public int[] Shape => GetShape();

        internal ActivationFunctions ActivationFunction;
        internal List<List<NeuronHolder>> Neurons;


        public double[] Execute(double[] input) => Execute(input, out _,  out _);

        public double[] Execute(double[] input, out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronActivations)
        {
            networkExecutionValues = new List<NeuronExecutionValues[]>();
            neuronActivations = new List<double[]>()
            {
                input
            };

            for (int i = 0; i < Length; i++)
            {

                int layerLength = Neurons[i].Count;
                neuronActivations.Add(new double[layerLength]);
                networkExecutionValues.Add(new NeuronExecutionValues[layerLength]);

                for (int j = 0; j < layerLength; j++)
                {
                    neuronActivations[i + 1][j] = Neurons[i][j].Execute(neuronActivations, ActivationFunction, out NeuronExecutionValues neuronExecutionValues);
                    networkExecutionValues[i][j] = neuronExecutionValues;
                }
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        internal List<List<NeuronHolder>> GetSupervisedLearningGradients(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, bool deleteMemoryAfterward = true)
        {
            List<double[]> outputs = new List<double[]>();
            List<List<NeuronExecutionValues[]>> networkExecutionsValues = new List<List<NeuronExecutionValues[]>>();
            List<List<double[]>> networkExecutionsNeuronOutputs = new List<List<double[]>>();
            List<double[]> costGradients = new List<double[]>();

            int tSCount = X.Count;
            for (int i = 0; i < tSCount; i++)
            {
                double[] currentNetworkOutput;
                outputs.Add(currentNetworkOutput = Execute(X[i], out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronOutputs));

                networkExecutionsValues.Add(networkExecutionValues);
                networkExecutionsNeuronOutputs.Add(neuronOutputs);

                costGradients.Add(Derivatives.DerivativeOf(currentNetworkOutput, y[i], costFunction));
            }

            var output = GetGradients(costGradients, networkExecutionsValues, networkExecutionsNeuronOutputs);

            if (deleteMemoryAfterward)
                DeleteMemory();

            return output;
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients"></param>
        /// <param name="executionValues">3D Grid in which time is the highest dimension and then layers and finally neurons</param>
        /// <param name="neuronActivations"></param>
        /// <returns></returns>
        internal List<List<NeuronHolder>> GetGradients(List<double[]> costGradients, List<List<NeuronExecutionValues[]>> executionValues, List<List<double[]>> neuronActivations)
        {
            List<List<NeuronHolder>> output = new List<List<NeuronHolder>>();
            List<List<List<double>>> neuronOutputGradientsGrid = ValueGeneration.GetTemporalNetworkCostGrid(costGradients, InputLength, Shape);
            int tSCount = costGradients.Count;

            for (int layerI = Neurons.Count - 1; layerI >= 0; layerI--)
            {
                output.Add(new List<NeuronHolder>());
                for (int neuronI = 0; neuronI < Neurons[layerI].Count; neuronI++)
                {
                    List<NeuronExecutionValues> neuronExecutionValues = new List<NeuronExecutionValues>();
                    for (int t = 0; t < tSCount; t++)
                        neuronExecutionValues.Add(executionValues[t][layerI][neuronI]);

                    NeuronHolder cNeuron = Neurons[layerI][neuronI];

                    output[layerI].Add(cNeuron.GetGradients(neuronOutputGradientsGrid[layerI][neuronI], neuronActivations, neuronExecutionValues, ActivationFunction, out List<double[]> connectionsGradients));

                    // Update neuronOutputGradientsGrid
                    for (int t = 0; t < tSCount; t++)
                        for (int i = 0; i < cNeuron.Connections.Length; i++)
                        {
                            Point connectedNeuronPos = cNeuron.Connections.ConnectedNeuronsPos[i];
                            neuronOutputGradientsGrid[connectedNeuronPos.X][connectedNeuronPos.Y][t] -= connectionsGradients[t][i];
                        }
                }
            }
            return output;
        }

        internal void SubtractGrads(List<List<NeuronHolder>> gradients, double learningRate)
        {
            for (int i = 0; i < Neurons.Count; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].SubtractGrads(gradients[i][j], learningRate);
        }

        /// <summary>
        /// For proper training and only if training use only after each train step, it isn't neccesary to use after each train step
        /// </summary>
        public void DeleteMemory()
        {
            for (int i = 0; i < Length; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].DeleteMemory();
        }

        private int[] GetShape()
        {
            int[] shape = new int[Length];
            for (int i = 0; i < Neurons.Count; i++)
                shape[i] = Neurons[i].Count;
            return shape;
        }
    }
}
