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
            Bias = 0.0;
            ForgetWeight = 0.0;
            StoreSigmoidWeight = 0.0;
            OutputWeight = 0.0;
        }

        internal double Execute(List<double[]> previousLayerActivations, out NeuronExecutionValues neuronExecutionVals)
        {
            neuronExecutionVals = new NeuronExecutionValues(NeuronHolder.NeuronTypes.LSTM)
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients"></param>
        /// <param name="executionValues">for proper training executionValues must have all the values since its memory was initialized</param>
        /// <returns></returns>
        internal LSTMNeuron GetGradients(List<double> costGradients, List<List<double[]>> networkNeuronOutputs, List<NeuronExecutionValues> executionValues, out List<double[]> connectionsActivationGradients)
        {
            int cCount = Connections.Length;

            LSTMNeuron output = new LSTMNeuron();
            output.Connections.ConnectedNeuronsPos = Connections.ConnectedNeuronsPos;
            output.Connections.Weights.AddRange(new double[Connections.Length]);

            connectionsActivationGradients = new List<double[]>();
            for (int i = 0; i < cCount; i++)
                connectionsActivationGradients.Add(new double[cCount]);


            int tSCount = costGradients.Count;

            double[] forgetWeightMultiplicationDerivatives = new double[tSCount];
            double[] forgetGateMultiplicationDerivatives = new double[tSCount];

            double[] storeGateSigmoidWeightMultiplicationDerivatives = new double[tSCount];
            double[] storeGateTanhWeightMultiplicationDerivatives = new double[tSCount];
            double[] storeGateMultiplicationDerivatives = new double[tSCount];
            double[] storeGateSumDerivative = new double[tSCount];

            double[] hiddenStateSigmoidDerivatives = new double[tSCount];
            double[] outputWeightMultiplicationDerivatives = new double[tSCount];
            double[] outputGateMultiplicationDerivatives = new double[tSCount];

            double[] outputCellStateTanhDerivatives = new double[tSCount];

            // Calculate Derivatives
            for (int t = 0; t < tSCount; t++)
            {
                double hiddenStateSigmoidDerivative = hiddenStateSigmoidDerivatives[t] = Derivatives.Sigmoid(executionValues[t].InitialHiddenStatePlusLinearFunction);

                // Forget Gate Derivatives
                double forgetGateWeightMultiplicationDerivative = forgetWeightMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(ForgetWeight, 0,
                                               executionValues[t].AfterForgetGateBeforeForgetWeightMultiplication, hiddenStateSigmoidDerivative);

                double forgetGateCellStateMultiplicationDerivative = forgetGateMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(executionValues[t].InitialCellState, storeGateSumDerivative[Math.Max(t - 1, 0)],
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

                storeGateSumDerivative[t] = Derivatives.Sum(forgetGateCellStateMultiplicationDerivative, storeGateMultiplicationDerivative);


                // Output Gate Derivatives
                double outputWeightMultiplicationDerivative = outputWeightMultiplicationDerivatives[t] = 
                    Derivatives.Multiplication(OutputWeight, 0,
                                               executionValues[t].AfterSigmoidBeforeWeightMultiplicationAtOutputGate, hiddenStateSigmoidDerivative);

                double outputCellStateTanhDerivative = outputCellStateTanhDerivatives[t] = Derivatives.Tanh(executionValues[t].OutputCellState);

                outputGateMultiplicationDerivatives[t] =
                    Derivatives.Multiplication
                    (
                        executionValues[t].AfterSigmoidAfterWeightMultiplicationAtOutputGate, outputWeightMultiplicationDerivative,
                        executionValues[t].AfterTanhOutputGate, outputCellStateTanhDerivative
                    );
            }

            // Calculate Gradients
            double previousHiddenStateGradient, previousCellStateGradient = previousHiddenStateGradient = 0;
            for (int t = tSCount - 1; t >= 0; t--)
            {
                costGradients[t] += previousHiddenStateGradient;

                costGradients[t] *= outputGateMultiplicationDerivatives[t];

                double cellStateGradient = outputCellStateTanhDerivatives[t] * costGradients[t] + previousCellStateGradient;


                double storeGateGradient = cellStateGradient *= storeGateSumDerivative[t];

                storeGateGradient *= storeGateMultiplicationDerivatives[t];

                double storeGateSigmoidWeightMultiplicationGradient = storeGateGradient * storeGateSigmoidWeightMultiplicationDerivatives[t];
                output.StoreSigmoidWeight += storeGateSigmoidWeightMultiplicationGradient;

                double storeGateTanhWeightMultiplicationGradient = storeGateGradient * storeGateTanhWeightMultiplicationDerivatives[t];
                output.StoreTanhWeight += storeGateTanhWeightMultiplicationGradient;

                double forgetGateGradient = previousCellStateGradient = cellStateGradient *= forgetGateMultiplicationDerivatives[t];

                double forgetWeightMultiplicationGradient = forgetGateGradient * forgetWeightMultiplicationDerivatives[t];
                output.ForgetWeight += forgetWeightMultiplicationGradient;

                // if GradientLearning doesn't work add not output gates gradients to hidden state gradients
                double outputGateWeightMultiplicationGradient = costGradients[t] *= outputWeightMultiplicationDerivatives[t];
                output.OutputWeight += outputGateWeightMultiplicationGradient;

                costGradients[t] *= hiddenStateSigmoidDerivatives[t];


                // calculate linear function derivative so gradient can pass through linear function + hiddenState
                double linearFunctionDerivative = 0;
                for (int i = 0; i < cCount; i++)
                {
                    Point currentConnectedPos = Connections.ConnectedNeuronsPos[i];
                    linearFunctionDerivative += networkNeuronOutputs[t][currentConnectedPos.X][currentConnectedPos.Y];
                }

                previousHiddenStateGradient = costGradients[t] *= linearFunctionDerivative + outputGateMultiplicationDerivatives[t - 1];

                for (int i = 0; i < cCount; i++)
                {
                    Point currentConnectedPos = Connections.ConnectedNeuronsPos[i];
                    output.Connections.Weights[t] += networkNeuronOutputs[t][currentConnectedPos.X][currentConnectedPos.Y] * costGradients[t];
                    connectionsActivationGradients[t][i] -= Connections.Weights[i] * costGradients[t];
                }
            }
            return output;
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
