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
        public NeuronType neuronType { get; private set; }
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
                        Output = activation,
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

        internal NeuronHolder GetGradients(List<double> costGradients, List<List<double[]>> previousOutputs, List<NeuronValues> neuronExecutionValues, Activation.ActivationFunctions activationFunction, out List<double[]> previousOutputsGradients)
        {
            NeuronHolder output = new NeuronHolder()
            {
                neuronType = neuronType,
            };
            previousOutputsGradients = new List<double[]>();
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    output.Neuron = Neuron.GetGradients(costGradients, neuronExecutionValues, previousOutputs, activationFunction);
                    break;
                case NeuronType.LSTM:
                    output.LSTMNeuron = LSTMNeuron.GetGradients(costGradients, previousOutputs, neuronExecutionValues, out previousOutputsGradients);
                    break;
                default:
                    throw new NotImplementedException();
            }
            return output;
        }

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

        internal void Evolve(double maxVariation, double mutationChance)
        {
            switch (neuronType)
            {
                case NeuronType.Neuron:
                    Neuron.Evolve(maxVariation, mutationChance);
                    break;
                case NeuronType.LSTM:
                    LSTMNeuron.Evolve(maxVariation, mutationChance);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

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
