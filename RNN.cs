using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeatNetwork.NetworkFiles;
using NeatNetwork.Libraries;
using static NeatNetwork.Libraries.Activation;

namespace NeatNetwork
{
    public class RNN
    {
        public int Length => Neurons.Count;
        public int InputLength => Neurons[0][0].Connections.Length;

        /// <summary>
        /// Doesn't include input
        /// </summary>
        public int[] Shape => GetShape();

        internal ActivationFunctions ActivationFunction;
        internal List<List<NeuronHolder>> Neurons;

        internal double InitialMaxMutationValue;
        internal List<List<double>> MaxMutationGrid;
        internal double MaxWeight;
        internal double MinWeight;
        internal double WeightClosestTo0;
        internal double NewBiasValue;
        internal double NewNeuronChance;
        internal double NewLayerChance;
        internal double FieldMaxMutation;
        internal double MaxMutationOfFieldMaxMutation;
        internal double MaxMutationOfMutationValueOfFieldMaxMutation;
        internal double MutationChance;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="shape">Includes input layer</param>
        /// <param name="layerTypes">Doesn't include input layer</param>
        /// <param name="activationFunction"></param>
        /// <param name="startingBias"></param>
        /// <param name="minWeight"></param>
        /// <param name="maxWeight"></param>
        /// <param name="weightClosestTo0"></param>
        public RNN(int[] shape, NeuronHolder.NeuronTypes[] layerTypes, ActivationFunctions activationFunction, double startingBias = 1, double minWeight = -1.5, double maxWeight = 1.5, double weightClosestTo0 = .37,
            double newNeuronChance = .05, double newLayerChance = .02, double mutationChance = .2, 
            double fieldMaxMutation = .27, double maxMutationOfFieldMaxMutation = .03, double maxMutationOfMutationValueOfFieldMaxMutation = .01)
        {
            ActivationFunction = activationFunction;
            Neurons = new List<List<NeuronHolder>>();
            for (int i = 1; i < shape.Length; i++)
            {
                Neurons.Add(new List<NeuronHolder>());
                for (int j = 0; j < shape[i]; j++)
                    Neurons[i - 1].Add(new NeuronHolder(layerTypes[i - 1], i, shape[i - 1], startingBias, maxWeight, minWeight, weightClosestTo0));
            }

            NewBiasValue = startingBias;
            MaxWeight = maxWeight;
            MinWeight = minWeight;
            WeightClosestTo0 = weightClosestTo0;

            NewNeuronChance = newNeuronChance;
            NewLayerChance = newLayerChance;
            MutationChance = mutationChance;
            FieldMaxMutation = fieldMaxMutation;
            MaxMutationOfFieldMaxMutation = maxMutationOfFieldMaxMutation;
            MaxMutationOfMutationValueOfFieldMaxMutation = maxMutationOfMutationValueOfFieldMaxMutation;
        }

        public double[] Execute(double[] input) => Execute(input, out _,  out _);

        public double[] Execute(double[] input, out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronActivations)
        {
            networkExecutionValues = new List<NeuronExecutionValues[]>();
            neuronActivations = new List<double[]>()
            {
                input
            };

            for (int i = 0; i < Length; i++)
            {

                int layerLength = Neurons[i].Count;
                neuronActivations.Add(new double[layerLength]);
                networkExecutionValues.Add(new NeuronExecutionValues[layerLength]);

                for (int j = 0; j < layerLength; j++)
                {
                    neuronActivations[i + 1][j] = Neurons[i][j].Execute(neuronActivations, ActivationFunction, out NeuronExecutionValues neuronExecutionValues);
                    networkExecutionValues[i][j] = neuronExecutionValues;
                }
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        #region Gradient Learning

        public void SupervisedLearningBatch(List<List<double[]>> X, List<List<double[]>> y, double batchSize, Cost.CostFunctions costFunction, double learningRate)
        {
            List<List<List<NeuronHolder>>> gradients = new List<List<List<NeuronHolder>>>();
            X = new List<List<double[]>>(X.ToArray());
            y = new List<List<double[]>>(y.ToArray());

            batchSize = Math.Abs(batchSize);
            batchSize *= 1 * Convert.ToInt16(batchSize > 1) + X.Count * Convert.ToInt16(batchSize <= 1);
            batchSize = Math.Ceiling(batchSize);


            Random r = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < batchSize; i++)
            {
                int dataI = r.Next(X.Count);
                gradients.Add(GetSupervisedLearningGradients(X[dataI], y[dataI], costFunction));

                X.RemoveAt(dataI);
                y.RemoveAt(dataI);
            }

            SubtractGrads(gradients, learningRate);
        }

        public void TrainBySupervisedLearning(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, double learningRate) =>
            SubtractGrads(GetSupervisedLearningGradients(X, y, costFunction), learningRate);

        internal List<List<NeuronHolder>> GetSupervisedLearningGradients(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, bool deleteMemoryBeforeAndAfter = true)
        {
            List<double[]> outputs = new List<double[]>();
            List<List<NeuronExecutionValues[]>> networkExecutionsValues = new List<List<NeuronExecutionValues[]>>();
            List<List<double[]>> networkExecutionsNeuronOutputs = new List<List<double[]>>();
            List<double[]> costGradients = new List<double[]>();

            if (deleteMemoryBeforeAndAfter)
                DeleteMemory();

            int tSCount = X.Count;
            for (int i = 0; i < tSCount; i++)
            {
                double[] currentNetworkOutput;
                outputs.Add(currentNetworkOutput = Execute(X[i], out List<NeuronExecutionValues[]> networkExecutionValues, out List<double[]> neuronOutputs));

                networkExecutionsValues.Add(networkExecutionValues);
                networkExecutionsNeuronOutputs.Add(neuronOutputs);

                costGradients.Add(Derivatives.DerivativeOf(currentNetworkOutput, y[i], costFunction));
            }

            var output = GetGradients(costGradients, networkExecutionsValues, networkExecutionsNeuronOutputs);

            if (deleteMemoryBeforeAndAfter)
                DeleteMemory();

            return output;
        }

        /// <param name="executionValues">3D Grid in which time is the highest dimension and then layersStrs and finally neurons</param>
        internal List<List<NeuronHolder>> GetGradients(List<double[]> costGradients, List<List<NeuronExecutionValues[]>> executionValues, List<List<double[]>> neuronActivations) =>
            GetGradients(costGradients, executionValues, neuronActivations, out _);


        /// <summary>
        /// 
        /// </summary>
        /// <param name="costGradients"></param>
        /// <param name="executionValues">3D Grid in which time is the highest dimension and then layersStrs and finally neurons</param>
        /// <param name="neuronActivations"></param>
        /// <param name="inputGradients"></param>
        /// <returns></returns>
        internal List<List<NeuronHolder>> GetGradients(List<double[]> costGradients, List<List<NeuronExecutionValues[]>> executionValues, List<List<double[]>> neuronActivations, out List<List<double>> inputGradients)
        {
            List<List<NeuronHolder>> output = new List<List<NeuronHolder>>();
            List<List<List<double>>> neuronOutputGradientsGrid = ValueGeneration.GetTemporalNetworkCostGrid(costGradients, InputLength, Shape);
            int tSCount = costGradients.Count;
            int lastLayerI = Neurons.Count - 1;

            for (int layerI = lastLayerI; layerI >= 0; layerI--)
            {
                output.Add(new List<NeuronHolder>());
                for (int neuronI = 0; neuronI < Neurons[layerI].Count; neuronI++)
                {
                    List<NeuronExecutionValues> neuronExecutionValues = new List<NeuronExecutionValues>();
                    for (int t = 0; t < tSCount; t++)
                        neuronExecutionValues.Add(executionValues[t][layerI][neuronI]);

                    NeuronHolder cNeuron = Neurons[layerI][neuronI];

                    output[lastLayerI - layerI].Add(cNeuron.GetGradients(neuronOutputGradientsGrid[layerI + 1][neuronI], neuronActivations, neuronExecutionValues, ActivationFunction, out List<double[]> connectionsGradients));

                    // Update neuronOutputGradientsGrid
                    for (int t = 0; t < tSCount; t++)
                        for (int i = 0; i < cNeuron.Connections.Length; i++)
                        {
                            Point connectedNeuronPos = cNeuron.Connections.ConnectedNeuronsPos[i];
                            neuronOutputGradientsGrid[connectedNeuronPos.X][connectedNeuronPos.Y][t] -= connectionsGradients[t][i];
                        }
                }
            }
            output.Reverse();
            inputGradients = neuronOutputGradientsGrid[0];
            return output;
        }

        internal void SubtractGrads(List<List<List<NeuronHolder>>> gradients, double learningRate)
        {
            foreach (var networkGradients in gradients)
                SubtractGrads(networkGradients, learningRate);
        }

        internal void SubtractGrads(List<List<NeuronHolder>> gradients, double learningRate)
        {
            for (int i = 0; i < Neurons.Count; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].SubtractGrads(gradients[i][j], learningRate);
        }

        #endregion

        #region Evolution learning

        public void AddInputNeuron()
        {

        }

        #endregion
        /// <summary>
        /// For proper training use only after each train step, it isn't neccesary to use after each train step
        /// </summary>
        public void DeleteMemory()
        {
            for (int i = 0; i < Length; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].DeleteMemory();
        }

        public override string ToString()
        {
            string output = string.Empty;
            foreach (var layer in Neurons)
            {
                foreach (var neuron in layer)
                {
                    output += neuron.ToString() + "/";
                }
                output = output.Remove(output.LastIndexOf('/'));
                output += "\n";
            }
            output = output.Remove(output.LastIndexOf('\n'));
            return output;
        }

        public RNN(string str)
        {
            Neurons = new List<List<NeuronHolder>>();
            string[] layersStrs = str.Split('\n');

            for (int i = 0; i < layersStrs.Length; i++)
            {
                Neurons.Add(new List<NeuronHolder>());

                string[] neuronsStrs = layersStrs[i].Split('/');
                foreach (var neuronStr in neuronsStrs)
                    Neurons[i].Add(new NeuronHolder(neuronStr));
            }
        }

        private int[] GetShape()
        {
            int[] shape = new int[Length];
            for (int i = 0; i < Neurons.Count; i++)
                shape[i] = Neurons[i].Count;
            return shape;
        }
    }
}
