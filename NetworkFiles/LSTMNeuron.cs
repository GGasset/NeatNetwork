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
        internal double StoreWeight;
        internal double OutputWeight;

        internal double Execute(List<double[]> previousLayerActivations, Activation.ActivationFunctions activationFunction, out NeuronValues neuronVals)
        {
            neuronVals = new NeuronValues(NeuronHolder.NeuronType.LSTM);

            double linearFunction = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                Point currentConnectedPos = weights.ConnectedNeuronsPos[i];
                linearFunction += previousLayerActivations[currentConnectedPos.X][currentConnectedPos.Y] * weights.Weights[i];
            }
            neuronVals.LinearFunction = linearFunction;
            HiddenState += linearFunction;
        }
    }
}
