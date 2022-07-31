using System;
using System.Collections.Generic;
using NeatNetwork.Libraries;

namespace NeatNetwork
{
    internal class NNEvolutionManager
    {
        internal List<NN> Networks;
        internal List<double> Scores { get; private set; }
        internal double MaxScore { get; private set; }
        internal double MinScore { get; private set; }
        internal int MaxScoredNetwork { get; set; }

        public NNEvolutionManager(int startingNetworkCount, int[] layerLengths, Activation.ActivationFunctions activation, double maxWeight = 1.5, double minWeight = -1.5, double weightClosestTo0 = 0.37, double startingBias = 1,
            double mutationChance = .1, double fieldMaxMutation = .04, double initialMaxMutationValue = .27, double newNeuronChance = .2, double newLayerChance = .05,
            double initialValueForMaxMutation = .27, double maxMutationOfMutationValues = .2, double maxMutationOfMutationValueOfMutationValues = .05)
        {
            Networks = new List<NN>();
            Scores = new List<double>();
            for (int i = 0; i < startingNetworkCount; i++)
            {
                Networks.Add(new NN(layerLengths, activation, maxWeight, minWeight, weightClosestTo0, startingBias, mutationChance, fieldMaxMutation, initialMaxMutationValue, newNeuronChance, newLayerChance
                    , initialValueForMaxMutation, maxMutationOfMutationValues, maxMutationOfMutationValueOfMutationValues));
            }
        }

        /// <summary>
        /// Function only applies to scored networks
        /// </summary>
        /// <param name="minChilds"></param>
        /// <param name="maxChilds"></param>
        public void HaveChild(int minChilds, int maxChilds)
        {
            (minChilds, maxChilds) = (Math.Max(minChilds, 0), Math.Max(maxChilds, 0));

            for (int i = 0; i < Scores.Count; i++)
            {
                double score = Scores[i];
                double currentChildCount = Math.Round((score - MinScore) / (MaxScore - MinScore) * (maxChilds - minChilds) + minChilds);
                currentChildCount = Math.Max(currentChildCount, minChilds);

                for (int j = 0; j < currentChildCount; j++)
                {
                    NN n = new NN(Networks[i].ToString());
                    n.Evolve();
                    Networks.Add(n);
                }
            }
        }

        public void DropWorstNetworks(double minMaxScorePercentageToSurvive, double maxNetworksToBeDeletedPercentage = 99)
        {
            minMaxScorePercentageToSurvive /= 1 + 99 * Convert.ToInt32(minMaxScorePercentageToSurvive > 1);
            maxNetworksToBeDeletedPercentage /= 1 + 99 * Convert.ToInt32(maxNetworksToBeDeletedPercentage > 1);

            double maxNetworksToBeDeleted = Math.Round(maxNetworksToBeDeletedPercentage * Networks.Count);
            double minScore = Math.Round(minMaxScorePercentageToSurvive * MaxScore);

            int deletedNetworksCount = 0;
            MinScore = MaxScore;
            for (int i = 0; i < Scores.Count && deletedNetworksCount < maxNetworksToBeDeleted; i++)
            {
                double currentScore = Scores[i];
                if (currentScore < minScore)
                {
                    Scores.RemoveAt(i);
                    deletedNetworksCount++;
                    i--;
                }
                else
                    MinScore += (currentScore - minScore) * Convert.ToInt32(currentScore < MinScore);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="score">Don't use numbers with exponent for correct functioning.</param>
        public void SetFirstNetworkScore(double score)
        {
            MaxScore = MinScore = score;
            Scores.Add(score);
        }

        /// <summary>
        /// If is the first network to be scored use SetFirstNetworkScore() for correct fuctioning.
        /// </summary>
        /// <param name="score">Don't use numbers with exponent for correct functioning.</param>
        public void SetNextNetworkToBeScoredScore(double score)
        {
            MaxScore += (score - MaxScore) * Convert.ToInt32(score > MaxScore);
            MaxScoredNetwork += (Scores.Count - MaxScoredNetwork) * Convert.ToInt32(score > MaxScore);

            MinScore += (score - MinScore) * Convert.ToInt32(score < MinScore);

            Scores.Add(score);
        }

        public NN GetMaxScoredNetwork() => Networks[MaxScoredNetwork];

        public NN GetNextToScoreNetwork() => GetNextToScoreNetwork(out _);

        public NN GetNextToScoreNetwork(out int networkIndex)
        {
            networkIndex = Scores.Count;
            return Networks[Scores.Count];
        }

        public bool AreAllNetworksScored()
        {
            return Scores.Count == Networks.Count;
        }

        /// <summary>
        /// This can be used to add other learning techniques while in network life cycle
        /// </summary>
        /// <param name="n"></param>
        /// <param name="index"></param>
        public void SetNetwork(NN n, int index)
        {
            Networks[index] = n;
        }

        public void SetNextToScoreNetwork(NN n)
        {
            Networks[Scores.Count] = n;
        }

        public void SetLastScoredNetwork(NN n)
        {
            Networks[Scores.Count - 1] = n;
        }
    }
}
