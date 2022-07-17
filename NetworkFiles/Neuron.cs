using System;
using System.Drawing;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.Libraries;


namespace NeatNetwork.NetworkFiles
{
    public class Neuron
    {
        public int ConnectionsLength => connections.Length;
        internal NeuronConnectionsInfo connections;
        internal double bias;

        /// <param name="neuronLayerIndex">layer 0 is input layer</param>
        public Neuron(int neuronLayerIndex, double defaultBias, int previousLayerLenght, double maxWeight, double minWeight, double weightClosestTo0)
        {
            bias = defaultBias;
            connections = new NeuronConnectionsInfo();
            for (int i = 0; i < previousLayerLenght; i++)
            {
                connections.AddNewConnection(neuronLayerIndex - 1, i, maxWeight, minWeight, weightClosestTo0);
            }
        }

        internal double Execute(List<double[]> previousLayersActivations, Activation.ActivationFunctions activationFunction, out double linearFunction)
        {
            linearFunction = 0;
            for (int i = 0; i < connections.Length; i++)
            {
                Point currentConnectedPos = connections.ConnectedNeuronsPos[i];
                linearFunction += connections.Weights[i] * previousLayersActivations[currentConnectedPos.X][currentConnectedPos.Y];
            }

            double output = linearFunction + bias;
            output = Activation.Activate(output, activationFunction);
            return output;
        }


        #region Gradient learning

        internal GradientValues GetGradients(int layerIndex, int neuronIndex, double cost, List<double[]> linearFunctions, List<double[]> neuronOutputs, Activation.ActivationFunctions activation)
        {
            GradientValues output = new GradientValues();
            double activationDerivative = Derivatives.DerivativeOf(linearFunctions[layerIndex][neuronIndex], activation);
            double activationGradient = cost * activationDerivative;

            output.biasGradient = activationGradient;

            for (int i = 0; i < connections.Length; i++)
            {
                Point currentConnectionPos = connections.ConnectedNeuronsPos[i];
                output.weightGradients.Add(activationGradient * neuronOutputs[currentConnectionPos.X][currentConnectionPos.Y]);

                output.previousActivationGradientsPosition.Add(currentConnectionPos);
                output.previousActivationGradients.Add(activationGradient * connections.Weights[i]);
            }
            return output;
        }

        internal void SubtractGrads(GradientValues gradients, double learningRate)
        {
            bias -= gradients.biasGradient * learningRate;
            for (int i = 0; i < ConnectionsLength; i++)
                connections.Weights[i] -= gradients.weightGradients[i] * learningRate;
        }

        #endregion


    }
}
