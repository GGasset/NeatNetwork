using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeatNetwork.NetworkFiles
{
    public class NeuronValues
    {
        internal readonly NeuronHolder.NeuronType NeuronType;

        // All neuron types
        internal double Activation;
        internal double LinearFunction;

        // Recurrent and LSTM
        internal double HiddenState;

        // LSTM
        internal double CellState;
        internal double AfterForgetGateBeforeForgetWeight;
        internal double AfterForgetGateAfterForgetWeight;
        internal double AfterSigmoidStoreGateBeforeStoreWeight;
        internal double AfterSigmoidStoreGateAfterStoreWeight;
        internal double AfterTanhStoreGate;
        internal double AfterStoreGateMultiplication;

        internal NeuronValues(NeuronHolder.NeuronType neuronType)
        {
            NeuronType = neuronType;
            Activation = 0;
            LinearFunction = 0.0;
            HiddenState = 0.0;
            CellState = 0.0;
        }
    }
}
