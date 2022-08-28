using NeatNetwork.Libraries;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    public class NeuronHolder
    {
        public NeuronTypes NeuronType { get; private set; }
        internal Neuron Neuron;
        internal LSTMNeuron LSTMNeuron;
        internal NeuronConnectionsInfo Connections => GetNeuronConnectionsInfo();

        internal NeuronHolder()
        {

        }

        internal NeuronHolder(NeuronTypes neuronType, int layerIndex, int previousLayerLength, double defaultBias = 1, double maxWeight = 1.5, double minWeight = -1.5, double valueClosestTo0 = 0.37)
        {
            NeuronType = neuronType;
            switch (neuronType)
            {
                case NeuronTypes.Neuron:
                    Neuron = new Neuron(layerIndex, previousLayerLength, maxWeight, defaultBias, minWeight, valueClosestTo0);
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron = new LSTMNeuron(layerIndex, previousLayerLength, maxWeight, defaultBias, minWeight, valueClosestTo0);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

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
                    output.Neuron = Neuron.GetGradients(costGradients, neuronExecutionValues, previousOutputs, activationFunction, out previousOutputsGradients);
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
            switch (gradients.NeuronType)
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
                default:
                    throw new NotImplementedException();
            }
        }

        public NeuronHolder(string str)
        {
            str = str.Replace("NeuronType: ", "");
            string[] values = str.Split('&');
            NeuronType = (NeuronTypes)Enum.Parse(typeof(NeuronTypes), values[0]);
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    Neuron = new Neuron(values[1]);
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron = new LSTMNeuron(values[1]);
                    break;
                default:
                    throw new NotImplementedException();
            }
        }

        public override string ToString()
        {
            string output = string.Empty;
            output += $"NeuronType: {Enum.GetName(typeof(NeuronTypes), NeuronType)}&";
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    output += Neuron.ToString();
                    break;
                case NeuronTypes.LSTM:
                    output += LSTMNeuron.ToString();
                    break;
                default:
                    throw new NotImplementedException();
            }
            return output;
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

        internal NeuronExecutionValues GetRecurrentState()
        {
            NeuronExecutionValues output = new NeuronExecutionValues(NeuronType);
            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    break;
                case NeuronTypes.LSTM:
                    output.OutputCellState = LSTMNeuron.CellState;
                    output.OutputHiddenState = LSTMNeuron.HiddenState;
                    break;
                default:
                    throw new NotImplementedException();
            }
            return output;
        }

        internal void SetRecurentState(NeuronExecutionValues recurrentState)
        {
            if (recurrentState.NeuronType != NeuronType)
                throw new ArgumentException("Error while setting recurrent state, NeuronTypes doesn't match");

            switch (NeuronType)
            {
                case NeuronTypes.Neuron:
                    break;
                case NeuronTypes.LSTM:
                    LSTMNeuron.CellState = recurrentState.OutputCellState;
                    LSTMNeuron.HiddenState = recurrentState.OutputHiddenState;
                    break;
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
