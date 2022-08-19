using NeatNetwork.Libraries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    internal class NeuronHolder
    {
        public readonly NeuronType neuronType;
        internal Neuron Neuron;
        internal LSTMNeuron LSTMNeuron;

        internal double Execute(List<double[]> previousNeuronsActivations, Activation.ActivationFunctions activationFunction, out NeuronValues ExecutionValues)
        {
            double activation;
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    activation = Neuron.Execute(previousNeuronsActivations, activationFunction, out double linearFunction);
                    ExecutionValues = new NeuronValues(neuronType)
                    {
                        LinearFunction = linearFunction,
                        Activation = activation,
                    };
                    break;
                case NeuronType.LSTM:
                    activation = LSTMNeuron.Execute(previousNeuronsActivations, out ExecutionValues);
                    break;
                default:
                    throw new NotImplementedException();
            }
            return activation;
        }

        #region Gradient learning

        internal void SubtractGrads(NeuronHolder gradients, double learningRate)
        {
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    Neuron.SubtractGrads(gradients.Neuron, learningRate);
                    break;
                case NeuronType.LSTM:
                    LSTMNeuron.SubtractGrads(gradients.LSTMNeuron, learningRate);
                    break;
                case NeuronType.Recurrent:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException();
            }
        }

        #endregion Gradient learning

        #region Evolution learning

        internal void AddConnection(int layerIndex, int neuronIndex, double weight)
        {
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    Neuron.Connections.AddNewConnection(layerIndex, neuronIndex, weight);
                    break;
                case NeuronType.LSTM:
                    LSTMNeuron.Connections.AddNewConnection(layerIndex, neuronIndex, weight);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        #endregion

        internal void DeleteMemory()
        {
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    break;
                case NeuronType.LSTM:
                    LSTMNeuron.DeleteMemory();
                    break;
                case NeuronType.Recurrent:
                    throw new NotImplementedException();
                default:
                    throw new NotImplementedException();
            }
        }

        public enum NeuronType
        {
            Neuron,
            LSTM,
            Recurrent
        }
    }
}
