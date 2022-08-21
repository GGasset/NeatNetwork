using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;
using NeatNetwork.NetworkFiles;
using static NeatNetwork.Libraries.ValueGeneration;

namespace NeatNetwork.NetworkFiles
{
    internal class LSTMNeuron
    {
        internal double CellState;
        internal double HiddenState;

        internal NeuronConnectionsInfo Connections;
        internal double Bias;

        internal double ForgetWeight;
        internal double StoreSigmoidWeight;
        internal double StoreTanhWeight;
        internal double OutputWeight;

        internal LSTMNeuron()
        {
            Connections = new NeuronConnectionsInfo();
        }

        internal double Execute(List<double[]> previousLayerActivations, out NeuronValues neuronExecutionVals)
        {
            neuronExecutionVals = new NeuronValues(NeuronHolder.NeuronType.LSTM)
            {
                InitialCellState = CellState,
                InitialHiddenState = HiddenState,
            };

            double linearFunction = Bias;
            for (int i = 0; i < Connections.Length; i++)
            {
                Point connectedPos = Connections.ConnectedNeuronsPos[i];
                linearFunction += previousLayerActivations[connectedPos.X][connectedPos.Y] * Connections.Weights[i];
            }
            neuronExecutionVals.LinearFunction = linearFunction;

            HiddenState += linearFunction;
            neuronExecutionVals.InitialHiddenStatePlusLinearFunction = HiddenState;

            double hiddenStateSigmoid = Activation.Sigmoid(HiddenState);

            double forgetGate = hiddenStateSigmoid;
            neuronExecutionVals.AfterForgetGateBeforeForgetWeightMultiplication = forgetGate;

            forgetGate *= ForgetWeight;
            neuronExecutionVals.AfterForgetGateSigmoidAfterForgetWeightMultiplication = forgetGate;

            CellState *= forgetGate;
            neuronExecutionVals.AfterForgetGateMultiplication = CellState;

            double storeGateSigmoidPath = hiddenStateSigmoid;
            neuronExecutionVals.AfterSigmoidStoreGateBeforeStoreWeightMultiplication = storeGateSigmoidPath;
            
            storeGateSigmoidPath *= StoreSigmoidWeight;
            neuronExecutionVals.AfterSigmoidStoreGateAfterStoreWeightMultiplication = storeGateSigmoidPath;

            double storeGateTanhPath = Activation.Tanh(HiddenState);
            neuronExecutionVals.AfterTanhStoreGateBeforeWeightMultiplication = storeGateTanhPath;

            storeGateTanhPath *= StoreTanhWeight;
            neuronExecutionVals.AfterTanhStoreGateAfterWeightMultiplication = storeGateTanhPath;

            double storeGate = storeGateSigmoidPath * storeGateTanhPath;
            neuronExecutionVals.AfterStoreGateMultiplication = storeGate;

            CellState += storeGate;

            double outputGateSigmoidPath = hiddenStateSigmoid;
            neuronExecutionVals.AfterSigmoidBeforeWeightMultiplicationAtOutputGate = outputGateSigmoidPath;

            outputGateSigmoidPath *= OutputWeight;
            neuronExecutionVals.AfterSigmoidAfterWeightMultiplicationAtOutputGate = outputGateSigmoidPath;

            double outputCellStateTanh = Activation.Tanh(CellState);
            neuronExecutionVals.AfterTanhOutputGate = outputCellStateTanh;

            HiddenState = outputGateSigmoidPath * outputCellStateTanh;

            neuronExecutionVals.OutputHiddenState = HiddenState;
            neuronExecutionVals.OutputCellState = CellState;

            neuronExecutionVals.Output = HiddenState;

            return HiddenState;
        }

        #region Gradient learning

        internal LSTMNeuron GetGradients(double[] costGradients, NeuronValues[] executionValues)
        {
            LSTMNeuron output = new LSTMNeuron();

            int tSCount = costGradients.Length;
            double[] cellStateDerivatives = new double[tSCount];

            double[] forgetWeightMultiplicationDerivatives = new double[tSCount];
            double[] forgetGateMultiplicationDerivatives = new double[tSCount];

            double[] storeGateSigmoidWeightMultiplicationDerivatives = new double[tSCount];
            double[] storeGateTanhWeightMultiplicationDerivatives = new double[tSCount];
            double[] storeGateMultiplicationDerivatives = new double[tSCount];

            double[] hiddenStateSigmoidDerivatives = new double[tSCount];
            double[] outputWeightMultiplicationDerivatives = new double[tSCount];
            double[] outputGateMultiplicationDerivatives = new double[tSCount];

            for (int t = 0; t < tSCount; t++)
            {
                double hiddenStateSigmoidDerivative = hiddenStateSigmoidDerivatives[t] = Derivatives.Sigmoid(executionValues[t].InitialHiddenStatePlusLinearFunction);

                // Forget Gate Derivatives
                double forgetGateWeightMultiplicationDerivative = forgetWeightMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(ForgetWeight, 0,
                                               executionValues[t].AfterForgetGateBeforeForgetWeightMultiplication, hiddenStateSigmoidDerivative);

                double forgetGateCellStateMultiplicationDerivative = forgetGateMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(executionValues[t].InitialCellState, cellStateDerivatives[Math.Max(t - 1, 0)],
                                               executionValues[t].AfterForgetGateSigmoidAfterForgetWeightMultiplication, forgetGateWeightMultiplicationDerivative);


                // Store Gate Derivatives
                double storeGateSigmoidWeightMultiplicationDerivative = storeGateSigmoidWeightMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(StoreSigmoidWeight, 0,
                                               executionValues[t].AfterSigmoidStoreGateBeforeStoreWeightMultiplication, hiddenStateSigmoidDerivative);

                double hiddenStateTanhDerivative = Derivatives.Tanh(executionValues[t].InitialHiddenStatePlusLinearFunction);
                double storeGateTanhWeightMultiplicationDerivative = storeGateTanhWeightMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(StoreTanhWeight, 0,
                                               executionValues[t].AfterTanhStoreGateBeforeWeightMultiplication, hiddenStateTanhDerivative);

                double storeGateMultiplicationDerivative = storeGateMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(executionValues[t].AfterSigmoidStoreGateAfterStoreWeightMultiplication, storeGateSigmoidWeightMultiplicationDerivative,
                                               executionValues[t].AfterTanhStoreGateAfterWeightMultiplication, storeGateTanhWeightMultiplicationDerivative);

                cellStateDerivatives[t] = Derivatives.Sum(forgetGateCellStateMultiplicationDerivative, storeGateMultiplicationDerivative);
                

                // Output Gate Derivatives
            }
        }
        
        internal void SubtractGrads(LSTMNeuron gradients, double learningRate)
        {
            Bias -= gradients.Bias * learningRate;
            Connections.SubtractGrads(Connections, learningRate);

            ForgetWeight -= gradients.ForgetWeight * learningRate;
            StoreSigmoidWeight -= gradients.StoreSigmoidWeight * learningRate;
            StoreTanhWeight -= gradients.StoreTanhWeight * learningRate;
            OutputWeight -= gradients.OutputWeight * learningRate;
        }

        #endregion

        #region Evolution learning

        internal void Evolve(double maxVariation, double mutationChance)
        {
            ForgetWeight += EvolveValue(maxVariation, mutationChance);
            StoreSigmoidWeight += EvolveValue(maxVariation, mutationChance);
            StoreTanhWeight += EvolveValue(maxVariation, mutationChance);
            OutputWeight += EvolveValue(maxVariation, mutationChance);

            Bias += EvolveValue(maxVariation, mutationChance);
            Connections.Evolve(maxVariation, mutationChance);
        }

        #endregion

        internal void DeleteMemory()
        {
            HiddenState = 0;
            CellState = 0;
        }
    }
}
