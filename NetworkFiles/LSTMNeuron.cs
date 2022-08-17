using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;
using NeatNetwork.NetworkFiles;

namespace NeatNetwork.NetworkFiles
{
    internal class LSTMNeuron
    {
        internal double CellState;
        internal double HiddenState;

        internal NeuronConnectionsInfo weights;

        internal double ForgetWeight;
        internal double StoreSigmoidWeight;
        internal double StoreTanhWeight;
        internal double OutputWeight;

        internal double Execute(List<double[]> previousLayerActivations, out NeuronValues neuronVals)
        {
            neuronVals = new NeuronValues(NeuronHolder.NeuronType.LSTM)
            {
                InitialCellState = CellState,
                InitialHiddenState = HiddenState,
            };

            double linearFunction = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                Point currentConnectedPos = weights.ConnectedNeuronsPos[i];
                linearFunction += previousLayerActivations[currentConnectedPos.X][currentConnectedPos.Y] * weights.Weights[i];
            }
            neuronVals.LinearFunction = linearFunction;

            double hiddenStateSigmoid = Activation.Sigmoid(HiddenState);

            double forgetGate = hiddenStateSigmoid;
            neuronVals.AfterForgetGateBeforeForgetWeightMultiplication = forgetGate;

            forgetGate *= ForgetWeight;
            neuronVals.AfterForgetGateSigmoidAfterForgetWeightMultiplication = forgetGate;

            CellState *= forgetGate;
            neuronVals.AfterForgetGateMultiplication = CellState;

            double storeGateSigmoidPath = hiddenStateSigmoid;

        }
    }
}
