﻿using NeatNetwork.Libraries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    internal class NeuronHolder
    {
        public NeuronTypes NeuronType { get; private set; }
        internal Neuron Neuron;
        internal LSTMNeuron LSTMNeuron;
        internal NeuronConnectionsInfo Connections => GetNeuronConnectionsInfo();

        internal double Execute(List<double[]> previousNeuronsActivations, Activation.ActivationFunctions activationFunction, out NeuronExecutionValues ExecutionValues)
        {
            double activation;
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    activation = Neuron.Execute(previousNeuronsActivations, activationFunction, out double linearFunction);
                    ExecutionValues = new NeuronExecutionValues(NeuronType)
                    {
                        LinearFunction = linearFunction,
                        Output = activation,
                    };
                    break;
                case NeuronTypes.LSTM:
                    activation = LSTMNeuron.Execute(previousNeuronsActivations, out ExecutionValues);
                    break;
                default:
                    throw new NotImplementedException();
            }
            return activation;
        }

        #region Gradient learning

        internal NeuronHolder GetGradients(List<double> costGradients, List<List<double[]>> previousOutputs, List<NeuronExecutionValues> neuronExecutionValues, Activation.ActivationFunctions activationFunction, out List<double[]> previousOutputsGradients)
        {
            NeuronHolder output = new NeuronHolder()
            {
                NeuronType = NeuronType,
            };
            previousOutputsGradients = new List<double[]>();
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    output.Neuron = Neuron.GetGradients(costGradients, neuronExecutionValues, previousOutputs, activationFunction);
                    break;
                case NeuronTypes.LSTM:
                    output.LSTMNeuron = LSTMNeuron.GetGradients(costGradients, previousOutputs, neuronExecutionValues, out previousOutputsGradients);
                    break;
                default:
                    throw new NotImplementedException();
            }
            return output;
        }

        internal void SubtractGrads(NeuronHolder gradients, double learningRate)
        {
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    Neuron.SubtractGrads(gradients.Neuron, learningRate);
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron.SubtractGrads(gradients.LSTMNeuron, learningRate);
                    break;
                case NeuronTypes.Recurrent:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException();
            }
        }

        #endregion Gradient learning

        #region Evolution learning

        internal void Evolve(double maxVariation, double mutationChance)
        {
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    Neuron.Evolve(maxVariation, mutationChance);
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron.Evolve(maxVariation, mutationChance);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        internal void AddConnection(int layerIndex, int neuronIndex, double weight)
        {
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    Neuron.Connections.AddNewConnection(layerIndex, neuronIndex, weight);
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron.Connections.AddNewConnection(layerIndex, neuronIndex, weight);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        #endregion

        internal void DeleteMemory()
        {
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron.DeleteMemory();
                    break;
                case NeuronTypes.Recurrent:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException();
            }
        }

        private NeuronConnectionsInfo GetNeuronConnectionsInfo()
        {
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    return Neuron.Connections;
                case NeuronTypes.LSTM:
                    return LSTMNeuron.Connections;
                default:
                    throw new NotImplementedException();
            }
        }

        public enum NeuronTypes
        {
            Neuron,
            LSTM,
            Recurrent
        }
    }
}
