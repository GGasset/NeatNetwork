﻿using System;
using System.Collections.Generic;
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

        internal List<List<NeuronHolder>> GetGradients(List<double[]> costGradients, List<List<NeuronExecutionValues[]>> executionValues, List<List<double[]>> neuronActivations)
        {
            List<List<List<double>>> executionGradients = ValueGeneration.GetTemporalNetworkCostGrid(costGradients, InputLength, Shape);
            int tSCount = costGradients.Count;

            for (int layerI = Neurons.Count - 1; layerI >= 0; layerI--)
            {
                for (int neuronI = 0; neuronI < Neurons[layerI].Count; neuronI++)
                {
                    Neurons[layerI][neuronI].GetGradients(executionGradients[layerI][neuronI], neuronActivations, , ActivationFunction, out List<double[]> )
                }
            }
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
