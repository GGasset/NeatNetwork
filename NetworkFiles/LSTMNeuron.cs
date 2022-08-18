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

        internal NeuronConnectionsInfo Weights;
        internal double Bias;

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

            double linearFunction = Bias;
            for (int i = 0; i < Weights.Length; i++)
            {
                Point connectedPos = Weights.ConnectedNeuronsPos[i];
                linearFunction += previousLayerActivations[connectedPos.X][connectedPos.Y] * Weights.Weights[i];
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
            neuronVals.AfterSigmoidStoreGateBeforeStoreWeightMultiplication = storeGateSigmoidPath;
            
            storeGateSigmoidPath *= StoreSigmoidWeight;
            neuronVals.AfterSigmoidStoreGateAfterStoreWeightMultiplication = storeGateSigmoidPath;

            double storeGateTanhPath = Activation.Tanh(HiddenState);
            neuronVals.AfterTanhStoreGateBeforeWeightMultiplication = storeGateTanhPath;

            storeGateTanhPath *= StoreTanhWeight;
            neuronVals.AfterTanhStoreGateAfterWeightMultiplication = storeGateTanhPath;

            double storeGate = storeGateSigmoidPath * storeGateTanhPath;
            neuronVals.AfterStoreGateMultiplication = storeGate;

            CellState += storeGate;

            double outputGateSigmoidPath = hiddenStateSigmoid;
            neuronVals.AfterSigmoidBeforeWeightMultiplicationAtOutputGate = outputGateSigmoidPath;

            outputGateSigmoidPath *= OutputWeight;
            neuronVals.AfterSigmoidAfterWeightMultiplicationAtOutputGate = outputGateSigmoidPath;

            double outputCellStateTanh = Activation.Tanh(CellState);
            neuronVals.AfterTanhOutputGate = outputCellStateTanh;

            HiddenState = outputGateSigmoidPath * outputCellStateTanh;

            neuronVals.OutputHiddenState = HiddenState;
            neuronVals.OutputCellState = CellState;

            neuronVals.Activation = HiddenState;

            return HiddenState;
        }

        internal void DeleteMemory()
        {
            HiddenState = 0;
            CellState = 0;
        }

        internal void SubtractGrads(LSTMNeuron gradients, double learningRate)
        {
            Bias -= gradients.Bias * learningRate;
            Weights.SubtractGrads(Weights, learningRate);

            ForgetWeight -= gradients.ForgetWeight * learningRate;
            StoreSigmoidWeight -= gradients.StoreSigmoidWeight * learningRate;
            StoreTanhWeight -= gradients.StoreTanhWeight * learningRate;
            OutputWeight -= gradients.OutputWeight * learningRate;
        }
    }
}
