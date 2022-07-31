using System;
using System.Collections.Generic;
using System.Drawing;
using NeatNetwork.NetworkFiles;
using static NeatNetwork.Libraries.ValueGeneration;
using NeatNetwork.Libraries;

namespace NeatNetwork
{
    public class NN
    {
        internal Activation.ActivationFunctions ActivationFunction;
        /// <summary>
        /// Input layer isn't instatiated
        /// </summary>
        internal List<List<Neuron>> Neurons;
        internal int InputLength;
        public int LayerCount => Neurons.Count;

        internal double InitialMaxMutationValue;
        internal List<List<double>> MaxMutationGrid;
        internal double MaxWeight;
        internal double MinWeight;
        internal double WeightClosestTo0;
        internal double NewBiasValue;
        internal double NewNeuronChance;
        internal double NewLayerChance;
        internal double FieldMaxMutation;
        internal double MaxMutationOfMutationValues;
        internal double MaxMutationOfMutationValueOfMutationValues;
        internal double MutationChance;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="layerLengths">Layer 0 in input layer and last layer is output layer</param>
        /// <param name="weightClosestTo0">If both max/min weight are positive or negative it will become useless</param>
        public NN(int[] layerLengths, Activation.ActivationFunctions activation, double maxWeight = 1.5, double minWeight = -1.5, double weightClosestTo0 = 0.37, double startingBias = 1,
            double mutationChance = .1, double fieldMaxMutation = .04, double initialMaxMutationValue = .27, double newNeuronChance = .2, double newLayerChance = .05,
            double initialValueForMaxMutation = .27, double maxMutationOfMutationValues = .2, double maxMutationOfMutationValueOfMutationValues = .05)
        {
            Neurons = new List<List<Neuron>>();
            MaxMutationGrid = new List<List<double>>();

            for (int i = 1; i < layerLengths.Length; i++)
            {
                Neurons.Add(new List<Neuron>());
                MaxMutationGrid.Add(new List<double>());

                for (int j = 0; j < Math.Abs(layerLengths[i]); j++)
                {
                    Neuron newNeuron = new Neuron(i, startingBias, layerLengths[i - 1], maxWeight, minWeight, weightClosestTo0);
                    Neurons[i - 1].Add(newNeuron);
                    MaxMutationGrid[i - 1].Add(initialValueForMaxMutation);
                }
            }

            this.InputLength = layerLengths[0];
            this.ActivationFunction = activation;
            this.MaxWeight = maxWeight;
            this.MinWeight = minWeight;
            this.WeightClosestTo0 = weightClosestTo0;
            this.NewBiasValue = startingBias;
            this.InitialMaxMutationValue = initialMaxMutationValue;
            this.MaxMutationOfMutationValues = maxMutationOfMutationValues;
            this.MaxMutationOfMutationValueOfMutationValues = maxMutationOfMutationValueOfMutationValues;
            this.MutationChance = mutationChance;
            this.FieldMaxMutation = fieldMaxMutation;
            this.NewNeuronChance = newNeuronChance;
            this.NewLayerChance = newLayerChance;
        }

        public double[] Execute(double[] input) => Execute(input, out _, out _);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="neuronActivations">first array corresponds to input</param>
        /// <param name="linearFunctions">first array corresponds to the next layer of input layer</param>
        /// <returns></returns>
        public double[] Execute(double[] input, out List<double[]> linearFunctions, out List<double[]> neuronActivations)
        {
            neuronActivations = new List<double[]>
            {
                input
            };

            linearFunctions = new List<double[]>();
            for (int i = 0; i < Neurons.Count; i++)
            {
                int layerLength = Neurons[i].Count;
                double[] layerOutput = new double[layerLength];
                double[] layerLinears = new double[layerLength];
                for (int j = 0; j < layerLength; j++)
                {
                    layerOutput[j] = Neurons[i][j].Execute(neuronActivations, ActivationFunction, out double linear);
                    layerLinears[j] = linear;
                }
                linearFunctions.Add(layerLinears);
                neuronActivations.Add(layerOutput);
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        public new string ToString()
        {
            string str = "";

            str += $"{InitialMaxMutationValue}\n{MaxWeight}\n{MinWeight}\n{WeightClosestTo0}\n{NewBiasValue}\n{NewNeuronChance}\n{NewLayerChance}\n{FieldMaxMutation}\n{MaxMutationOfMutationValues}\n" +
                $"{MaxMutationOfMutationValueOfMutationValues}\n{MutationChance}\n{Enum.GetName(typeof(Activation.ActivationFunctions), ActivationFunction)}\n";
            str += "HIHI\n";
            foreach (var layerMaxMutation in MaxMutationGrid)
            {
                foreach (var neuronMaxMutation in layerMaxMutation)
                {
                    str += $"{neuronMaxMutation},";
                }
                str += "\n-\n";
            }
            str += "HIHI\n";
            foreach (var layer in Neurons)
            {
                foreach (var neuron in layer)
                {
                    str += neuron.ToString() + "_";
                }
                str += "\n-\n";
            }
            return str;
        }

        public NN(string str)
        {
            string[] principalStrs = str.Split(new string[] { "HIHI\n" }, StringSplitOptions.RemoveEmptyEntries);

            string[] fieldsStrs = principalStrs[0].Split(new string[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);
            InitialMaxMutationValue = Convert.ToDouble(fieldsStrs[0]);
            MaxWeight = Convert.ToDouble(fieldsStrs[1]);
            MinWeight = Convert.ToDouble(fieldsStrs[2]);
            WeightClosestTo0 = Convert.ToDouble(fieldsStrs[3]);
            NewBiasValue = Convert.ToDouble(fieldsStrs[4]);
            NewNeuronChance = Convert.ToDouble(fieldsStrs[5]);
            NewLayerChance = Convert.ToDouble(fieldsStrs[6]);
            FieldMaxMutation = Convert.ToDouble(fieldsStrs[7]);
            MaxMutationOfMutationValues = Convert.ToDouble(fieldsStrs[8]);
            MaxMutationOfMutationValueOfMutationValues = Convert.ToDouble(fieldsStrs[9]);
            MutationChance = Convert.ToDouble(fieldsStrs[10]);
            ActivationFunction = (Activation.ActivationFunctions)Enum.Parse(typeof(Activation.ActivationFunctions), fieldsStrs[11]);

            MaxMutationGrid = new List<List<double>>();
            string[] maxMutationsLayersStrs = principalStrs[1].Split(new string[] { "\n-\n" }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < maxMutationsLayersStrs.Length; i++)
            {
                MaxMutationGrid.Add(new List<double>());
                string[] currentLayerNeuronsMaxMutationsStrs = maxMutationsLayersStrs[i].Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var neuronMaxMutationStr in currentLayerNeuronsMaxMutationsStrs)
                {
                    MaxMutationGrid[i].Add(Convert.ToDouble(neuronMaxMutationStr));
                }
            }

            Neurons = new List<List<Neuron>>();
            string[] layerStrs = principalStrs[2].Split(new string[] { "\n-\n" }, StringSplitOptions.RemoveEmptyEntries);
            for (int layerIndex = 0; layerIndex < layerStrs.Length; layerIndex++)
            {
                Neurons.Add(new List<Neuron>());
                string[] currentLayerNeuronsStrs = layerStrs[layerIndex].Split(new string[] { "_" }, StringSplitOptions.RemoveEmptyEntries);
                for (int neuronIndex = 0; neuronIndex < currentLayerNeuronsStrs.Length; neuronIndex++)
                {
                    Neurons[layerIndex].Add(new Neuron(currentLayerNeuronsStrs[neuronIndex]));
                }
            }
        }

        #region Gradient learning

        static int RandomI = int.MinValue / 2;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="X"></param>
        /// <param name="y"></param>
        /// <param name="costFunction">you must select a supervised learning cost function.</param>
        /// <returns>Mean cost</returns>
        public double SupervisedLearningBatch(List<double[]> X, List<double[]> y, int batchLength, Cost.CostFunctions costFunction, double learningRate) => SupervisedLearningBatch(X, y, batchLength, costFunction, learningRate, out _);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="X"></param>
        /// <param name="y"></param>
        /// <param name="costFunction">you must select a supervised learning cost function.</param>
        /// <returns>Mean cost</returns>
        public double SupervisedLearningBatch(List<double[]> X, List<double[]> y, int batchLength, Cost.CostFunctions costFunction, double learningRate, out List<double> meanCosts)
        {
            if (X.Count != y.Count)
                throw new IndexOutOfRangeException();

            List<List<GradientValues[]>> gradients = new List<List<GradientValues[]>>();
            meanCosts = new List<double>();
            double meanCost = 0;
            Random r = new Random(DateTime.Now.Millisecond + RandomI);
            RandomI++;
            for (int i = 0; i < batchLength; i++)
            {
                int trainingI = r.Next(X.Count);
                gradients.Add(GetSupervisedGradients(X[trainingI], y[trainingI], costFunction, out double executionMeanCost));
                meanCosts.Add(executionMeanCost);
                meanCost += executionMeanCost;
            }
            meanCost /= X.Count;

            foreach (var currentNetworkGradients in gradients)
            {
                SubtractGrads(currentNetworkGradients, learningRate);
            }

            return meanCost;
        }

        public List<GradientValues[]> GetSupervisedGradients(double[] X, double[] y, Cost.CostFunctions costFunction, out double meanCost)
        {
            double[] output = Execute(X, out List<double[]> neuronLinears, out List<double[]> neuronActivations);
            double[] costGradients = Derivatives.DerivativeOf(output, y, costFunction);
            meanCost = Cost.GetCost(output, y, costFunction);

            return GetGradients(neuronLinears, neuronActivations, costGradients);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="linearFunctions">This doesn't include input</param>
        /// <param name="neuronActivations">Includes input</param>
        /// <param name="costs"></param>
        /// <returns></returns>
        public List<GradientValues[]> GetGradients(List<double[]> linearFunctions, List<double[]> neuronActivations, double[] costs) => GetGradients(linearFunctions, neuronActivations, costs, out _);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="linearFunctions">Doesn't include input</param>
        /// <param name="neuronActivations">Includes input</param>
        /// <param name="costs"></param>
        /// <returns></returns>
        public List<GradientValues[]> GetGradients(List<double[]> linearFunctions, List<double[]> neuronActivations, double[] costs, out double[] inputCosts)
        {
            List<GradientValues[]> output = new List<GradientValues[]>();
            int layerCount = Neurons.Count;
            for (int i = 0; i < layerCount; i++)
            {
                int layerLength = Neurons[i].Count;
                output.Add(new GradientValues[layerLength]);
            }

            inputCosts = new double[neuronActivations[0].Length];
            List<double[]> costGrid = GetNeuronCostsGrid(costs);

            for (int layerIndex = Neurons.Count - 1; layerIndex >= 0; layerIndex--)
            {
                int layerLength = Neurons[layerIndex].Count;
                for (int neuronIndex = 0; neuronIndex < layerLength; neuronIndex++)
                {
                    double currentCost = costGrid[layerIndex + 1][neuronIndex];
                    GradientValues currentGradients = Neurons[layerIndex][neuronIndex].GetGradients(layerIndex, neuronIndex, currentCost, linearFunctions, neuronActivations, ActivationFunction);
                    output[layerIndex][neuronIndex] = currentGradients;

                    // update grid / set input costs
                    for (int connectionIndex = 0; connectionIndex < currentGradients.previousActivationGradients.Count; connectionIndex++)
                    {
                        Point connectedPos = currentGradients.previousActivationGradientsPosition[connectionIndex];
                        double currentConnectedGradient = currentGradients.previousActivationGradients[connectionIndex];

                        costGrid[connectedPos.X][connectedPos.Y] -= currentConnectedGradient;
                    }
                }
            }

            inputCosts = costGrid[0];
            return output;
        }

        public void SubtractGrads(List<GradientValues[]> gradients, double learningRate)
        {
            for (int i = 0; i < LayerCount; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].SubtractGrads(gradients[i][j], learningRate);
        }

        internal List<double[]> GetNeuronCostsGrid(double[] outputCosts)
        {
            List<double[]> output = new List<double[]>
            {
                new double[InputLength]
            };

            for (int i = 0; i < Neurons.Count; i++)
            {
                int layerLength = Neurons[i].Count;
                output.Add(new double[layerLength]);
            }

            int outputLayerLength = Neurons[Neurons.Count - 1].Count;
            for (int i = 0; i < outputLayerLength; i++)
            {
                // Corresponds to output layerMaxMutation counting with input layerMaxMutation
                output[Neurons.Count][i] = outputCosts[i];
            }

            return output;
        }

        #endregion

        #region Evolution learning

        internal void Evolve()
        {
            InitialMaxMutationValue += GetVariation(-MaxMutationOfMutationValues, MaxMutationOfMutationValues) * WillMutate(MutationChance);

            MaxWeight += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);
            MinWeight += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);
            WeightClosestTo0 += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            NewBiasValue += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            NewNeuronChance += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);
            NewLayerChance += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            FieldMaxMutation += GetVariation(-MaxMutationOfMutationValues, MaxMutationOfMutationValues) * WillMutate(MutationChance);
            MaxMutationOfMutationValues += GetVariation(-MaxMutationOfMutationValueOfMutationValues, MaxMutationOfMutationValueOfMutationValues) * WillMutate(MutationChance);
            MutationChance += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            for (int i = 0; i < Neurons.Count; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                {
                    Neurons[i][j].Evolve(MutationChance, MaxMutationGrid[i][j]);
                    MaxMutationGrid[i][j] += GetVariation(-MaxMutationOfMutationValues, MaxMutationOfMutationValues) * WillMutate(MutationChance);
                }

            int insertionIndex = new Random(RandomI++).Next(Neurons.Count - 1);
            if (WillMutate(NewNeuronChance) == 1 && Neurons.Count > 1)
                AddNewNeuron(insertionIndex);
            else if (WillMutate(NewLayerChance) == 1)
                AddNewLayer(insertionIndex, 1);
        }

        internal void AddNewLayer(int layerInsertionIndex, int layerLength) 
        {
            for (int i = layerInsertionIndex; i < Neurons.Count; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].Connections.AdjustToNewLayerBeingAdded(layerInsertionIndex, i == layerInsertionIndex, layerLength, this.MinWeight, this.MaxWeight, this.WeightClosestTo0);

            int previousLayerLength = layerInsertionIndex > 0 ? Neurons[layerInsertionIndex - 1].Count : InputLength;
            List<Neuron> layer = new List<Neuron>();
            List<double> layerMaxMutationGrid = new List<double>();
            for (int i = 0; i < layerLength; i++)
            {
                layer.Add(new Neuron(layerInsertionIndex + 1, NewBiasValue, previousLayerLength, MaxWeight, MinWeight, WeightClosestTo0));
                layerMaxMutationGrid.Add(InitialMaxMutationValue);
            }

            Neurons.Insert(layerInsertionIndex, layer);
            MaxMutationGrid.Insert(layerInsertionIndex, layerMaxMutationGrid);
        }

        internal void AddNewNeuron(int layerInsertionIndex)
        {
            int previousLayerLength = layerInsertionIndex > 0 ? Neurons[layerInsertionIndex - 1].Count : InputLength;

            Neurons[layerInsertionIndex].Add(new Neuron(layerInsertionIndex + 1, NewBiasValue, previousLayerLength, MaxWeight, MinWeight, WeightClosestTo0));
            MaxMutationGrid[layerInsertionIndex].Add(InitialMaxMutationValue);
        }

        #endregion

    }
}
