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
        public int ConnectionsLength => Connections.Length;
        internal NeuronConnectionsInfo Connections;
        internal double Bias;

        public Neuron(NeuronConnectionsInfo connections, double bias)
        {
            this.Connections = connections;
            this.Bias = bias;
        }

        /// <param name="neuronLayerIndex">layer 0 is input layer</param>
        public Neuron(int neuronLayerIndex, double defaultBias, int previousLayerLenght, double maxWeight, double minWeight, double weightClosestTo0)
        {
            Bias = defaultBias;
            Connections = new NeuronConnectionsInfo();
            for (int i = 0; i < previousLayerLenght; i++)
            {
                Connections.AddNewConnection(neuronLayerIndex - 1, i, maxWeight, minWeight, weightClosestTo0);
            }
        }

        internal double Execute(List<double[]> previousLayersActivations, Activation.ActivationFunctions activationFunction, out double linearFunction)
        {
            linearFunction = 0;
            for (int i = 0; i < Connections.Length; i++)
            {
                Point currentConnectedPos = Connections.ConnectedNeuronsPos[i];
                linearFunction += Connections.Weights[i] * previousLayersActivations[currentConnectedPos.X][currentConnectedPos.Y];
            }

            double output = linearFunction + Bias;
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

            for (int i = 0; i < Connections.Length; i++)
            {
                Point currentConnectionPos = Connections.ConnectedNeuronsPos[i];
                output.weightGradients.Add(activationGradient * neuronOutputs[currentConnectionPos.X][currentConnectionPos.Y]);

                output.previousActivationGradientsPosition.Add(currentConnectionPos);
                output.previousActivationGradients.Add(activationGradient * Connections.Weights[i]);
            }
            return output;
        }

        internal void SubtractGrads(GradientValues gradients, double learningRate)
        {
            Bias -= gradients.biasGradient * learningRate;
            Connections.SubtractGrads(gradients.weightGradients, learningRate);
        }

        internal void SubtractGrads(Neuron gradients, double learningRate)
        {
            Bias -= gradients.Bias * learningRate;
            Connections.SubtractGrads(gradients.Connections, learningRate);
        }

        #endregion

        #region Evolution learning

        internal void Evolve(double mutationChance, double maxMutation)
        {
            Connections.Evolve(mutationChance, maxMutation);

            Bias += ValueGeneration.GetVariation(-maxMutation, maxMutation) * ValueGeneration.WillMutate(mutationChance);
        }

        #endregion

        internal new string ToString()
        {
            string str = "";
            str += $"Bias: {Bias}^ Connections: {Connections.ToString()}";
            return str;
        }

        public Neuron(string str)
        {
            str = str.Replace("Bias: ", "").Replace(" Connections: ", "");
            string[] fieldStrs = str.Split(new char[] { '^' });
            Bias = Convert.ToDouble(fieldStrs[0]);
            Connections = new NeuronConnectionsInfo(fieldStrs[1]);
        }
    }
}
