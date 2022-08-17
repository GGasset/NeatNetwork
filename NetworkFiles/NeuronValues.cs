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
        internal double AfterForgetGateBeforeForgetWeightMultiplication;
        internal double AfterForgetGateAfterForgetWeightMultiplication;
        internal double AfterForgetGateMultiplication;

        // Store gate
        internal double AfterSigmoidStoreGateBeforeStoreWeightMultiplication;
        internal double AfterSigmoidStoreGateAfterStoreWeightMultiplication;
        internal double AfterTanhStoreGate;
        internal double AfterStoreGateMultiplication;

        // Output gate
        internal double AfterSigmoidBeforeWeightMultiplicationAtOutputGate;
        internal double AfterSigmoidAfterWeightMultiplicationAtOutputGate;
        internal double AfterTanhOutputGate;


        internal NeuronValues(NeuronHolder.NeuronType neuronType)
        {
            NeuronType = neuronType;
        }
    }
}
