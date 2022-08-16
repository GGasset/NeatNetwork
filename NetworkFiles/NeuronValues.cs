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
        internal double InitialHiddenState;
        internal double OutputHiddenState;

        // LSTM
        internal double InitialCellState;
        internal double OutputCellState;

        //Forget gate
        internal double AfterForgetGateBeforeForgetWeight;
        internal double AfterForgetGateAfterForgetWeight;
        internal double AfterForgetWeightMultiplication;

        // Store gat
        internal double AfterSigmoidStoreGateBeforeStoreWeight;
        internal double AfterSigmoidStoreGateAfterStoreWeight;
        internal double AfterTanhStoreGate;
        internal double AfterStoreGateMultiplication;
        internal double AfterStoreGateAddition;

        // Output gate

        internal NeuronValues(NeuronHolder.NeuronType neuronType)
        {
            NeuronType = neuronType;
        }
    }
}
