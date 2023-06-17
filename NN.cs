using NeatNetwork.Libraries;
using NeatNetwork.NetworkFiles;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using static NeatNetwork.Libraries.ValueGeneration;

namespace NeatNetwork
{
    public class NN
    {
        internal Activation.ActivationFunctions ActivationFunction;

        /// <summary>
        /// Input layer isn't instantiated
        /// </summary>
        internal List<List<Neuron>> Neurons;

        public readonly int InputLength;
        public int LayerCount => Neurons.Count;
        public int[] Shape => GetShape();

        internal double InitialMaxMutationValue;
        internal List<List<double>> MaxMutationGrid;
        internal double MaxWeight;
        internal double MinWeight;
        internal double WeightClosestTo0;
        internal double NewBiasValue;
        internal double NewNeuronChance;
        internal double NewLayerChance;
        internal double FieldMaxMutation;
        internal double MaxMutationOFieldMaxMutation;
        internal double MaxMutationOfMutationValueOfFieldMaxMutation;
        internal double MutationChance;

        /// <summary>
        ///
        /// </summary>
        /// <param name="shape">Includes input layer</param>
        /// <param name="weightClosestTo0">If both max/min weight are positive or negative it will become useless</param>
        public NN(int[] shape, Activation.ActivationFunctions activation, double maxWeight = 1.5, double minWeight = -1.5, double weightClosestTo0 = 0.37, double startingBias = 1,
            double mutationChance = .1, double fieldMaxMutation = .04, double initialMaxMutationValue = .27, double newNeuronChance = .2, double newLayerChance = .05,
            double initialValueForMaxMutation = .27, double maxMutationOfMutationValues = .2, double maxMutationOfMutationValueOfMutationValues = .05)
        {
            Neurons = new List<List<Neuron>>();
            MaxMutationGrid = new List<List<double>>();

            List<Task<List<Neuron>>> layersTasks = new List<Task<List<Neuron>>>();
            AsyncLayerInstantiator[] layerInstantiators = new AsyncLayerInstantiator[shape.Length];
            for (int i = 1; i < shape.Length; i++)
            {
                MaxMutationGrid.Add(new List<double>());

                int layerLength = shape[i];

                layerInstantiators[i] = new AsyncLayerInstantiator(i, layerLength, shape[i - 1], startingBias, maxWeight, minWeight, weightClosestTo0);
                layersTasks.Add(layerInstantiators[i].InstantiateLayerAsync());

                for (int j = 0; j < layerLength; j++)
                {
                    MaxMutationGrid[i - 1].Add(initialValueForMaxMutation);
                }
            }

            foreach (var task in layersTasks)
            {
                task.Wait();
            }

            foreach (var task in layersTasks)
            {
                Neurons.Add(task.Result);
            }

            this.InputLength = shape[0];
            this.ActivationFunction = activation;
            this.MaxWeight = maxWeight;
            this.MinWeight = minWeight;
            this.WeightClosestTo0 = weightClosestTo0;
            this.NewBiasValue = startingBias;
            this.InitialMaxMutationValue = initialMaxMutationValue;
            this.MaxMutationOFieldMaxMutation = maxMutationOfMutationValues;
            this.MaxMutationOfMutationValueOfFieldMaxMutation = maxMutationOfMutationValueOfMutationValues;
            this.MutationChance = mutationChance;
            this.FieldMaxMutation = fieldMaxMutation;
            this.NewNeuronChance = newNeuronChance;
            this.NewLayerChance = newLayerChance;
        }

        private class AsyncLayerInstantiator
        {
            private readonly int LayerI, LayerLength, PreviousLayerLength;
            private readonly double Bias, MaxWeight, MinWeight, WeightClosestTo0;

            internal AsyncLayerInstantiator(int layerI, int layerLength, int previousLayerLength, double bias, double maxWeight, double minWeight, double weightClosestTo0)
            {
                LayerI = layerI;
                LayerLength = layerLength;
                PreviousLayerLength = previousLayerLength;
                Bias = bias;
                MaxWeight = maxWeight;
                MinWeight = minWeight;
                WeightClosestTo0 = weightClosestTo0;
            }

            internal Task<List<Neuron>> InstantiateLayerAsync() => Task.Run(() => InstantiateLayer(LayerI, LayerLength, PreviousLayerLength, Bias, MaxWeight, MinWeight, WeightClosestTo0));
        }

        private static List<Neuron> InstantiateLayer(int layerI, int layerLength, int previousLayerLength, double startingBias, double maxWeight, double minWeight, double weightClosestTo0)
        {
            List<Neuron> output = new List<Neuron>();
            List<Task<Neuron>> neuronTasks = new List<Task<Neuron>>();
            for (int i = 0; i < layerLength; i++)
            {
                neuronTasks.Add(Task.Run(() => new Neuron(layerI, previousLayerLength, startingBias, maxWeight, minWeight, weightClosestTo0)));
            }

            foreach (var task in neuronTasks)
            {
                task.Wait();
            }

            foreach (var neuronTask in neuronTasks)
            {
                output.Add(neuronTask.Result);
            }

            return output;
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
                (double[] layerOutput, double[] layerLinears) = ExecuteLayer(i, neuronActivations);

                linearFunctions.Add(layerLinears);
                neuronActivations.Add(layerOutput);
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="input"></param>
        /// <param name="layerI">Inclusive, layerI is executed, input layer not considerated</param>
        /// <returns></returns>
        public double[] ExecuteUpToLayer(double[] input, int layerI)
        {
            List<double[]> neuronActivations = new List<double[]>()
            {
                input
            };

            for (int i = 0; i <= layerI; i++)
            {
                (double[] layerOutput, _) = ExecuteLayer(i, neuronActivations);
                neuronActivations.Add(layerOutput);
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index">this layer (input layer is not considered a layer) is set as if this layer output is the parameter layerActivations</param>
        /// <param name="layerActivations"></param>
        /// <returns></returns>
        public double[] ExecuteFromLayer(int index, double[] layerActivations)
        {
            List<double[]> neuronActivations = new List<double[]>()
            {
                new double[InputLength]
            };

            for (int i = 0; i < index; i++)
            {
                neuronActivations.Add(new double[Neurons[i].Count]);
            }

            neuronActivations.Add(layerActivations);

            for (int i = index + 1; i < Neurons.Count; i++)
            {
                (double[] layerOutput, _) = ExecuteLayer(i, neuronActivations);
                neuronActivations.Add(layerOutput);
            }

            return neuronActivations[neuronActivations.Count - 1];
        }

        private (double[] layerOutputs, double[] layerLinears) ExecuteLayer(int layerI, List<double[]> previousActivations)
        {
            int layerLength = Neurons[layerI].Count;
            double[] layerOutput = new double[layerLength];
            double[] layerLinears = new double[layerLength];

            List<Task<(double neuronActivation, double neuronLinear)>> executionTasks = new List<Task<(double, double)>>();
            AsyncNeuronExecutor[] asyncNeuronExecutors = new AsyncNeuronExecutor[layerLength];
            for (int j = 0; j < layerLength; j++)
            {
                asyncNeuronExecutors[j] = new AsyncNeuronExecutor(Neurons[layerI][j]);
                executionTasks.Add(asyncNeuronExecutors[j].ExecuteNeuronAsync(previousActivations, ActivationFunction));
            }

            foreach (var task in executionTasks)
            {
                task.Wait();
            }

            for (int i = 0; i < layerLength; i++)
                (layerOutput[i], layerLinears[i]) = executionTasks[i].Result;

            return (layerOutput, layerLinears);
        }

        private class AsyncNeuronExecutor
        {
            private readonly Neuron Neuron;

            internal AsyncNeuronExecutor(Neuron neuron)
            {
                Neuron = neuron;
            }

            internal Task<(double neuronActivation, double neuronLinear)> ExecuteNeuronAsync(List<double[]> previousActivations, Activation.ActivationFunctions activationFunction) =>
                Task.Run(() => Neuron.Execute(previousActivations, activationFunction));
        }

        public override string ToString()
        {
            string str = "";

            str += $"{InitialMaxMutationValue}\n{MaxWeight}\n{MinWeight}\n{WeightClosestTo0}\n{NewBiasValue}\n{NewNeuronChance}\n{NewLayerChance}\n{FieldMaxMutation}\n{MaxMutationOFieldMaxMutation}\n" +
                $"{MaxMutationOfMutationValueOfFieldMaxMutation}\n{MutationChance}\n{Enum.GetName(typeof(Activation.ActivationFunctions), ActivationFunction)}\n";
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

        public NN(StreamReader file)
        {
            string currentLine;

            InitialMaxMutationValue = Convert.ToDouble(file.ReadLine());
            MaxWeight = Convert.ToDouble(file.ReadLine());
            MinWeight = Convert.ToDouble(file.ReadLine());
            WeightClosestTo0 = Convert.ToDouble(file.ReadLine());
            NewBiasValue = Convert.ToDouble(file.ReadLine());
            NewNeuronChance = Convert.ToDouble(file.ReadLine());
            NewLayerChance = Convert.ToDouble(file.ReadLine());
            FieldMaxMutation = Convert.ToDouble(file.ReadLine());
            MaxMutationOFieldMaxMutation = Convert.ToDouble(file.ReadLine());
            MaxMutationOfMutationValueOfFieldMaxMutation = Convert.ToDouble(file.ReadLine());
            MutationChance = Convert.ToDouble(file.ReadLine());
            ActivationFunction = (Activation.ActivationFunctions)Enum.Parse(typeof(Activation.ActivationFunctions), file.ReadLine());


            if (file.ReadLine() != "HIHI")
                throw new FormatException("The readed file is not from the app.");
            MaxMutationGrid = new List<List<double>>();


            string maxMutationGridStr = "";
            while ((currentLine = file.ReadLine()) != "HIHI")
            {
                maxMutationGridStr = currentLine + "\n";
            }

            MaxMutationGrid = InstantiateMaxMutationGrid(maxMutationGridStr);



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
            MaxMutationOFieldMaxMutation = Convert.ToDouble(fieldsStrs[8]);
            MaxMutationOfMutationValueOfFieldMaxMutation = Convert.ToDouble(fieldsStrs[9]);
            MutationChance = Convert.ToDouble(fieldsStrs[10]);
            ActivationFunction = (Activation.ActivationFunctions)Enum.Parse(typeof(Activation.ActivationFunctions), fieldsStrs[11]);

            MaxMutationGrid = InstantiateMaxMutationGrid(principalStrs[1]);

            string[] layerStrs = principalStrs[2].Split(new string[] { "\n-\n" }, StringSplitOptions.RemoveEmptyEntries);
            List<Task<List<Neuron>>> layerTasks = new List<Task<List<Neuron>>>();
            AsyncFromStringLayerInstantiator[] layerInstantiators = new AsyncFromStringLayerInstantiator[layerStrs.Length];
            for (int layerIndex = 0; layerIndex < layerStrs.Length; layerIndex++)
            {
                layerInstantiators[layerIndex] = new AsyncFromStringLayerInstantiator(layerStrs[layerIndex]);
                layerTasks.Add(layerInstantiators[layerIndex].InstantiateLayerFromStringAsync());
            }

            Neurons = new List<List<Neuron>>();
            foreach (var task in layerTasks)
            {
                task.Wait();
                Neurons.Add(task.Result);
            }
        }

        private static List<List<double>> InstantiateMaxMutationGrid(string str)
        {
            List<List<double>> maxMutationGrid = new List<List<double>>();
            string[] layers = str.Split(new string[] { "\n-\n" }, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < layers.Length; i++)
            {
                maxMutationGrid.Add(new List<double>());
                string[] currentLayerNeuronsMaxMutationsStrs = layers[i].Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                foreach (var neuronMaxMutationStr in currentLayerNeuronsMaxMutationsStrs)
                {
                    maxMutationGrid[i].Add(Convert.ToDouble(neuronMaxMutationStr));
                }
            }
            return maxMutationGrid;
        }

        private static List<Neuron> InstantiateLayer(string str)
        {
            string[] neuronsStrs = str.Split(new string[] { "_" }, StringSplitOptions.RemoveEmptyEntries);

            List<Task<Neuron>> neuronTasks = new List<Task<Neuron>>();
            AsyncFromStringNeuronInstantiator[] neuronInstantiators = new AsyncFromStringNeuronInstantiator[neuronsStrs.Length];

            for (int neuronIndex = 0; neuronIndex < neuronsStrs.Length; neuronIndex++)
            {
                neuronInstantiators[neuronIndex] = new AsyncFromStringNeuronInstantiator(neuronsStrs[neuronIndex]);
                neuronTasks.Add(neuronInstantiators[neuronIndex].InstatiateNeuronAsync());
            }

            List<Neuron> output = new List<Neuron>();
            foreach (var task in neuronTasks)
            {
                task.Wait();
                output.Add(task.Result);
            }

            return output;
        }

        private class AsyncFromStringLayerInstantiator
        {
            private readonly string str;

            internal AsyncFromStringLayerInstantiator(string str)
            {
                this.str = str;
            }

            internal Task<List<Neuron>> InstantiateLayerFromStringAsync() => Task.Run(() => InstantiateLayer(str));
        }

        private class AsyncFromStringNeuronInstantiator
        {
            private readonly string str;

            internal AsyncFromStringNeuronInstantiator(string str)
            {
                this.str = str;
            }

            internal Task<Neuron> InstatiateNeuronAsync() => Task.Run(() => new Neuron(str));
        }

        #region Gradient learning

        private static int RandomI = int.MinValue / 2;

        public double SupervisedTrain(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, double learningRate, double testSize = 0.2, int batchSize = 350, bool shuffleData = true)
        {
            if (X.Count != y.Count)
                throw new ArgumentOutOfRangeException("X - y", "X.Count is different than y.Count");

            if (shuffleData)
                (X, y) = DataManipulation.ShuffleData(X, y);

            ((List<double[]> trainX, List<double[]> trainY), (List<double[]> testX, List<double[]> testY)) = DataManipulation.SliceData(X, y, 1 - testSize);

            int batchCount = trainX.Count / batchSize;
            int lastBatchSize = trainX.Count % batchSize;

            for (int i = 0; i < batchCount; i++)
            {
                SupervisedLearningBatch(trainX, trainY, costFunction, learningRate, i * batchSize, (i + 1) * batchSize, out double currentMeanCost);
                Console.WriteLine($"{(i + 1) * batchSize}/{trainX.Count}");
            }
            SupervisedLearningBatch(trainX, trainY, costFunction, learningRate, trainX.Count - lastBatchSize, trainX.Count, out _);
            Console.WriteLine($"{trainX.Count}/{trainX.Count}");

            double meanCost = 0;
            for (int i = 0; i < testX.Count; i++)
            {
                var output = Execute(testX[i]);
                double cost = Cost.GetCost(output, testY[i], costFunction);
                meanCost += cost;
            }
            meanCost /= testX.Count;
            return meanCost;
        }

        public void SupervisedLearningBatch(List<double[]> X, List<double[]> y, Cost.CostFunctions costFunction, double learningRate, int startIndex, int exclusiveEndIndex, out double meanCost)
        {
            List<Task<(List<GradientValues[]>, double)>> gradientsTasks = new List<Task<(List<GradientValues[]>, double cost)>>();
            List<AsyncGradientsCalculator> asyncGradientsCalculators = new List<AsyncGradientsCalculator>();
            for (int i = startIndex; i < exclusiveEndIndex; i++)
            {
                asyncGradientsCalculators.Add(new AsyncGradientsCalculator(X[i], y[i], this));
                gradientsTasks.Add(asyncGradientsCalculators[i - startIndex].GetGradientsAsync(costFunction));
            }

            foreach (var task in gradientsTasks)
            {
                task.Wait();
            }

            meanCost = 0;
            foreach (var task in gradientsTasks)
            {
                (List<GradientValues[]> currentGradients, double currentCost) = task.Result;

                SubtractGrads(currentGradients, learningRate);
                meanCost += currentCost;
            }
            meanCost /= gradientsTasks.Count;
        }

        private class AsyncGradientsCalculator
        {
            private readonly double[] X, y;
            private readonly NN n;

            internal AsyncGradientsCalculator(double[] x, double[] y, NN n)
            {
                X = x;
                this.y = y;
                this.n = n;
            }

            internal Task<(List<GradientValues[]> gradients, double cost)> GetGradientsAsync(Cost.CostFunctions costFunction) =>
                Task.Run(() => n.GetSupervisedGradients(X, y, costFunction));
        }

        /// <summary>
        /// Supervised learning batch with data selected randomly
        /// </summary>
        /// <param name="X"></param>
        /// <param name="y"></param>
        /// <param name="costFunction">you must select a supervised learning cost function.</param>
        /// <returns>Mean cost</returns>
        public double SupervisedLearningBatch(List<double[]> X, List<double[]> y, int batchLength, Cost.CostFunctions costFunction, double learningRate) => SupervisedLearningBatch(X, y, batchLength, costFunction, learningRate, out _);

        /// <summary>
        /// Supervised learning batch with data selected randomly
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

        internal (List<GradientValues[]> gradients, double cost) GetSupervisedGradients(double[] X, double[] y, Cost.CostFunctions costFunction)
        {
            var gradients = GetSupervisedGradients(X, y, costFunction, out double meanCost);
            return (gradients, meanCost);
        }

        internal List<GradientValues[]> GetSupervisedGradients(double[] X, double[] y, Cost.CostFunctions costFunction, out double meanCost)
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

            List<double[]> costGrid = ValueGeneration.GetNetworkCostGrid(InputLength, Shape, costs);

            for (int layerIndex = Neurons.Count - 1; layerIndex >= 0; layerIndex--)
            {
                int layerLength = Neurons[layerIndex].Count;
                for (int neuronIndex = 0; neuronIndex < layerLength; neuronIndex++)
                {
                    double currentCost = costGrid[layerIndex + 1][neuronIndex];
                    GradientValues currentGradients = Neurons[layerIndex][neuronIndex].GetGradients(currentCost, linearFunctions[layerIndex][neuronIndex], neuronActivations, ActivationFunction);
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

        class AsyncNeuronGradientsCalculator
        {
            readonly Neuron Neuron;
            readonly double NeuronCost, NeuronLinear;
            readonly List<double[]> NeuronsActivations;

            internal AsyncNeuronGradientsCalculator(Neuron neuron, double neuronCost, double neuronLinear, List<double[]> neuronsActivations)
            {
                Neuron = neuron;
                NeuronCost = neuronCost;
                NeuronLinear = neuronLinear;
                NeuronsActivations = neuronsActivations;
            }

            internal Task<GradientValues> GetNeuronGradientsAsync(Activation.ActivationFunctions activationFunction)
                => Task.Run(() => Neuron.GetGradients(NeuronCost, NeuronLinear, NeuronsActivations, activationFunction));
        }

        public void SubtractGrads(List<GradientValues[]> gradients, double learningRate)
        {
            for (int i = 0; i < LayerCount; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                    Neurons[i][j].SubtractGrads(gradients[i][j], learningRate);
        }

        #endregion Gradient learning

        #region Evolution learning

        internal void Evolve()
        {
            InitialMaxMutationValue += GetVariation(-MaxMutationOFieldMaxMutation, MaxMutationOFieldMaxMutation) * WillMutate(MutationChance);

            MaxWeight += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);
            MinWeight += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);
            WeightClosestTo0 += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            NewBiasValue += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            NewNeuronChance += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);
            NewLayerChance += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            FieldMaxMutation += GetVariation(-MaxMutationOFieldMaxMutation, MaxMutationOFieldMaxMutation) * WillMutate(MutationChance);
            MaxMutationOFieldMaxMutation += GetVariation(-MaxMutationOfMutationValueOfFieldMaxMutation, MaxMutationOfMutationValueOfFieldMaxMutation) * WillMutate(MutationChance);
            MutationChance += GetVariation(-FieldMaxMutation, FieldMaxMutation) * WillMutate(MutationChance);

            for (int i = 0; i < Neurons.Count; i++)
                for (int j = 0; j < Neurons[i].Count; j++)
                {
                    Neurons[i][j].Evolve(MaxMutationGrid[i][j], MutationChance);
                    MaxMutationGrid[i][j] += GetVariation(-MaxMutationOFieldMaxMutation, MaxMutationOFieldMaxMutation) * WillMutate(MutationChance);
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
                layer.Add(new Neuron(layerInsertionIndex + 1, previousLayerLength, NewBiasValue, MaxWeight, MinWeight, WeightClosestTo0));
                layerMaxMutationGrid.Add(InitialMaxMutationValue);
            }

            Neurons.Insert(layerInsertionIndex, layer);
            MaxMutationGrid.Insert(layerInsertionIndex, layerMaxMutationGrid);
        }

        internal void AddNewNeuron(int layerInsertionIndex)
        {
            int previousLayerLength = layerInsertionIndex > 0 ? Neurons[layerInsertionIndex - 1].Count : InputLength;

            Neurons[layerInsertionIndex].Add(new Neuron(layerInsertionIndex + 1, previousLayerLength, NewBiasValue, MaxWeight, MinWeight, WeightClosestTo0));
            MaxMutationGrid[layerInsertionIndex].Add(InitialMaxMutationValue);
        }

        #endregion Evolution learning

        private int[] GetShape()
        {
            int[] output = new int[Neurons.Count];
            for (int i = 0; i < Neurons.Count; i++)
                output[i] = Neurons[i].Count;

            return output;
        }

        public NN Clone() => (NN)MemberwiseClone();
    }
}