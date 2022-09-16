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

        public Neuron()
        {
            this.Connections = new NeuronConnectionsInfo();
            this.Bias = 0;
        }

        /// <param name="neuronLayerIndex">layer 0 is input layer</param>
        public Neuron(int neuronLayerIndex, int previousLayerLenght, double defaultBias, double maxWeight, double minWeight, double weightClosestTo0)
        {
            Bias = defaultBias;
            Connections = new NeuronConnectionsInfo(neuronLayerIndex, previousLayerLenght, minWeight, maxWeight, weightClosestTo0);
            
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="previousLayersActivations"></param>
        /// <param name="activationFunction"></param>
        /// <returns>tuple that represents (neuronActivation, neuronLinear)</returns>
        internal (double, double) Execute(List<double[]> previousLayersActivations, Activation.ActivationFunctions activationFunction)
        {
            double neuronActivation = Execute(previousLayersActivations, activationFunction, out double neuronLinear);
            return (neuronActivation, neuronLinear);
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

        internal GradientValues GetGradients(double cost, double linearFunction, List<double[]> neuronActivations, Activation.ActivationFunctions activation)
        {
            GradientValues output = new GradientValues();
            double activationDerivative = Derivatives.DerivativeOf(linearFunction, activation);
            double activationGradient = cost * activationDerivative;

            output.biasGradient = activationGradient;

            for (int i = 0; i < Connections.Length; i++)
            {
                Point currentConnectionPos = Connections.ConnectedNeuronsPos[i];
                output.weightGradients.Add(activationGradient * neuronActivations[currentConnectionPos.X][currentConnectionPos.Y]);

                output.previousActivationGradientsPosition.Add(currentConnectionPos);
                output.previousActivationGradients.Add(activationGradient * Connections.Weights[i]);
            }
            return output;
        }

        /// <summary>
        /// Function only used for RNNs
        /// </summary>
        /// <returns></returns>
        internal Neuron GetGradients(List<double> costs, List<NeuronExecutionValues> executionValues, List<List<double[]>> neuronActivations, Activation.ActivationFunctions activationFunction, out List<double[]> previousNeuronsGradients)
        {
            int tCount = costs.Count;
            int cCount = Connections.Length;

            Neuron output = new Neuron()
            {
                Connections = Connections,
            };

            for (int i = 0; i < Connections.Length; i++)
                output.Connections.Weights[i] = 0;

            previousNeuronsGradients = new List<double[]>();
            for (int i = 0; i < tCount; i++)
                previousNeuronsGradients.Add(new double[cCount]);

            for (int t = tCount - 1; t >= 0; t--)
            {
                GradientValues gradients = GetGradients(costs[t], executionValues[t].LinearFunction, neuronActivations[t], activationFunction);
                output.Bias += gradients.biasGradient;

                for (int i = 0; i < cCount; i++)
                {
                    output.Connections.Weights[i] += gradients.weightGradients[i];
                    previousNeuronsGradients[t][i] -= gradients.previousActivationGradients[i];
                }
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

        internal void Evolve(double maxMutation, double maxVariation)
        {
            Connections.Evolve(maxVariation, maxMutation);

            Bias += ValueGeneration.EvolveValue(maxMutation, maxVariation);
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
