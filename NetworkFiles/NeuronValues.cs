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

        //Forget weight
        internal double AfterForgetGateBeforeForgetWeight;
        internal double AfterForgetGateAfterForgetWeight;
        internal double AfterForgetWeightMultiplication;

        // Store weight
        internal double AfterSigmoidStoreGateBeforeStoreWeight;
        internal double AfterSigmoidStoreGateAfterStoreWeight;
        internal double AfterTanhStoreGate;
        internal double AfterStoreGateMultiplication;
        internal double AfterStoreGateAddition;

        internal NeuronValues(NeuronHolder.NeuronType neuronType)
        {
            NeuronType = neuronType;
        }
    }
}
