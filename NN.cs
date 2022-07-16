using System;
using System.Collections.Generic;
using System.Drawing;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;

namespace NeatNetwork
{
    public class NN
    {
        internal Activation.ActivationFunctions Activation;
        /// <summary>
        /// Input layer isn't instatiated
        /// </summary>
        internal List<List<Neuron>> Neurons;
        internal int InputLength => Neurons[0][0].connections.Length;
        public int LayerCount => Neurons.Count;
        internal List<List<double>> MaxMutationGrid;
        internal double MaxMutationOfMutationValues;
        internal double MaxMutationOfMutationValueOfMutationValues;
        internal double MutationProbability;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layerLengths">Layer 0 in input layer and last layer is output layer</param>
        /// <param name="weightClosestTo0">If both max/min weight are positive or negative it will become useless</param>
        public NN(int[] layerLengths, Activation.ActivationFunctions activation, double maxWeight = 1.5, double minWeight = -1.5, double weightClosestTo0 = 0.37, double startingBias = 1, double mutationChance = .10, double initialValueForMaxMutation = .27, double maxMutationOfMutationValues = .2, double maxMutationOfMutationValueOfMutationValues = .05)
        {
            Neurons = new List<List<Neuron>>();
            MaxMutationGrid = new List<List<double>>();

            for (int i = 1; i < layerLengths.Length; i++)
            {
                Neurons.Add(new List<Neuron>());
                MaxMutationGrid.Add(new List<double>());

                for (int j = 0; j < Math.Abs(layerLengths[i]); j++)
                {
                    Neurons[i].Add(new Neuron(i, startingBias, layerLengths[i - 1], maxWeight, minWeight, weightClosestTo0));
                    MaxMutationGrid[i].Add(initialValueForMaxMutation);
                }
            }

            this.Activation = activation;
            this.MaxMutationOfMutationValues = maxMutationOfMutationValues;
            this.MaxMutationOfMutationValueOfMutationValues = maxMutationOfMutationValueOfMutationValues;
            MutationProbability = mutationChance;
        }

        internal double[] Execute(double[] input) => Execute(input, out _, out _);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="neuronOutputs">first array corresponds to input</param>
        /// <param name="linearFunctions">first array corresponds to the next layer of input layer</param>
        /// <returns></returns>
        internal double[] Execute(double[] input, out List<double[]> neuronOutputs, out List<double[]> linearFunctions)
        {
            neuronOutputs = new List<double[]>
            {
                input
            };

            linearFunctions = new List<double[]>();
            for (int i = 0; i < Neurons.Count; i++)
            {
                int layerLength = Neurons[i].Count;
                double[] layerOutput = new double[layerLength];
                double[] layerLinears = new double[layerLength];
                for (int j = 0; j < layerLength; j++)
                {
                    layerOutput[j] = Neurons[i][j].Execute(neuronOutputs, Activation, out double linear);
                    layerLinears[j] = linear;
                }
                linearFunctions.Add(layerLinears);
                neuronOutputs.Add(layerOutput);
            }

            return neuronOutputs[neuronOutputs.Count - 1];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="linearFunctions">This doesn't include input</param>
        /// <param name="neuronActivations">Includes input</param>
        /// <param name="costs"></param>
        /// <returns></returns>
        internal List<GradientValues[]> GetGradients(List<double[]> linearFunctions, List<double[]> neuronActivations, double[] costs) => GetGradients(linearFunctions, neuronActivations, costs, out _);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="linearFunctions">Doesn't include input</param>
        /// <param name="neuronActivations">Includes input</param>
        /// <param name="costs"></param>
        /// <returns></returns>
        internal List<GradientValues[]> GetGradients(List<double[]> linearFunctions, List<double[]> neuronActivations, double[] costs, out double[] inputCosts)
        {
            List<GradientValues[]> output = new List<GradientValues[]>();
            inputCosts = new double[neuronActivations[0].Length];
            List<double[]> costGrid = GetNeuronCostsGrid(costs);

            for (int i = Neurons.Count - 1; i >= 0; i--)
            {
                int layerLength = Neurons[i].Count;
                output.Add(new GradientValues[layerLength]);
                for (int j = 0; j < layerLength; j++)
                {
                    GradientValues currentGradients = Neurons[i][j].GetGradients(i, j, costGrid[i][j], linearFunctions, neuronActivations, Activation);
                    output[i][j] = currentGradients;

                    // update grid / set input costs
                    for (int k = 0; k < currentGradients.previousActivationGradients.Count; k++)
                    {
                        Point connectedPos = currentGradients.previousActivationGradientsPosition[k];
                        double currentConnectedGradient = currentGradients.previousActivationGradients[k];

                        costGrid[connectedPos.X][connectedPos.Y] -= currentConnectedGradient;
                        inputCosts[Math.Min(connectedPos.Y, inputCosts.Length - 1)] -= currentConnectedGradient * Convert.ToInt32(connectedPos.Y == 0);
                    }
                }
            }
            return output;
        }

        internal void SubtractGrads(List<List<GradientValues>> gradients)
        {
            for (int i = 0; i < LayerCount; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].SubtractGrads(gradients[i][j]);
        }

        internal List<double[]> GetNeuronCostsGrid(double[] outputCosts)
        {
            List<double[]> output = new List<double[]>();

            for (int i = 0; i < Neurons.Count - 1; i++)
            {
                int layerLength = Neurons[i].Count;
                output.Add(new double[layerLength]);
                for (int j = 0; j < layerLength; j++)
                {
                    output[i][j] = 0;
                }
            }

            int outputLayerLength = Neurons[Neurons.Count - 1].Count;
            output.Add(new double[outputLayerLength]);
            for (int i = 0; i < outputLayerLength; i++)
            {
                output[Neurons.Count - 1][i] = outputCosts[i];
            }

            return output;
        }
    }
}
