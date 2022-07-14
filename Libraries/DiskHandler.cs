using System.IO;
using System.Collections.Generic;

namespace NN.Libraries
{
    public static class DiskHandler
    {
        public static void SaveToDisk(string name, string folderPath, string str)
        {
            File.WriteAllText(GetPath(name, folderPath), str);
        }

        public static string ReadFromDisk(string fileName, string folderPath) => File.ReadAllText(GetPath(fileName, folderPath));

        public static void DeleteFile(string fileName, string folderPath)
        {
            File.Delete(GetPath(fileName, folderPath));
        }

        public static List<string> GetFilesInFolder(string folderPath)
        {
            return new List<string>(Directory.GetFiles(folderPath));
        }

        private static string GetPath(string fileName, string folderPath)
        {
            return $@"{fileName}\{folderPath}";
        }
    }
}
