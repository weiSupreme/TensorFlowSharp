using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Configuration;
using TensorFlow;
using ExampleCommon;
using Mono.Options;
using System.Reflection;
using System.Net;
using ICSharpCode.SharpZipLib.Tar;
using ICSharpCode.SharpZipLib.GZip;
using System.Drawing;
using System.Drawing.Imaging;

namespace ExampleObjectDetection
{
	class Program
	{

		private static double MIN_SCORE_FOR_OBJECT_HIGHLIGHTING = 0.5;

        static TFShape tfs = new TFShape(1, 320, 320,3);


        /// <param name="args"></param>
        static void Main (string [] args)
		{
            TFSessionOptions options = new TFSessionOptions();
            using (var graph = new TFGraph())
            {
                var model = File.ReadAllBytes("model/frozen_inference_graph.pb");
                graph.Import(new TFBuffer(model));
                using (var session = new TFSession(graph))
                {
                    string[] labels = File.ReadAllLines("labels.txt");
                    //while(true)
                    string[] files = Directory.GetFiles("t2", "*.*");
                    //while(true)
                    //FileStream fs = new FileStream("res.txt", FileMode.Create);
                    int cnt = 0;
                    foreach (string file in files)
                    {
                        //Console.WriteLine(file);
                        DateTime bft = DateTime.Now;
                        var tensor = Image2Tensor(file);
                        //DateTime bft = DateTime.Now;
                        //DateTime aft = DateTime.Now;
                        //break;
                        var runner = session.GetRunner();
                        runner
                            .AddInput(graph["image_tensor"][0], tensor)
                            .Fetch(
                            graph["detection_boxes"][0],
                            graph["detection_scores"][0],
                            graph["detection_classes"][0],
                            graph["num_detections"][0]);
                        var output = runner.Run();

                        var boxes = (float[,,])output[0].GetValue(jagged: false);
                        var scores = (float[,])output[1].GetValue(jagged: false);
                        var classes = (float[,])output[2].GetValue(jagged: false);
                        var num = (float[])output[3].GetValue(jagged: false);
                        DateTime aft = DateTime.Now;
                        if (num[0] > 7)
                        {
                            Dictionary<int[], string> dict = new Dictionary<int[], string>();
                            int num1 = CreateDict(boxes, classes, labels, ref dict);
                            foreach (int[] key in dict.Keys)
                                Console.WriteLine("box: {0} {1} {2} {3}, class: {4}", key[0], key[1], key[2], key[3], dict[key]);
                        }
                        Console.WriteLine("time: {0} ms",(aft - bft).TotalMilliseconds.ToString());
                        cnt++;
                    }
                }
            }
            Console.ReadLine();
        }

        static int cropTLX = 121;
        static int cropTLY = 105;
        private static int CreateDict(float[,,] boxes, float[,] classes, string[] labels, ref Dictionary<int[], string> dict)
        {
            int mean = 0;
            int num1 = 0;
            var y = boxes.GetLength(1);
            var z = boxes.GetLength(2);
            Dictionary<int[], string> dict1 = new Dictionary<int[], string>();
            Dictionary<int[], string> dict2 = new Dictionary<int[], string>();
            for (int j = 0; j < y; j++)
            {
                int[] box = new int[z];
                for (int k = 0; k < z; k++)
                {
                    var coord = boxes[0, j, k];
                    if (k % 2 == 0)
                    {
                        box[k] = (int)(coord * cropSize + cropTLY);
                        if (0 == k)
                            mean += box[0];
                    }
                    else
                        box[k] = (int)(coord * cropSize + cropTLX);
                }
                int value = Convert.ToInt32(classes[0, j]);
                dict.Add(box, labels[value-1]);
            }
            mean = (int)(mean / y);
            foreach (int[] key in dict.Keys)
            {
                if (mean > key[0])
                {
                    dict1.Add(key, dict[key]);
                    num1 += 1;
                }
                else
                {
                    dict2.Add(key, dict[key]);
                }
            }
            dict1 = dict1.OrderBy(kv => kv.Key[1]).ToDictionary(k => k.Key, v => v.Value);
            dict2 = dict2.OrderBy(kv => kv.Key[1]).ToDictionary(k => k.Key, v => v.Value);
            dict = dict1.Concat(dict2).ToDictionary(k => k.Key, v => v.Value);
            return num1;
        }

        private static void DrawBoxes (float [,,] boxes, float [,] scores, float [,] classes, string[] label, string inputFile, string outputFile, double minScore)
		{
			var x = boxes.GetLength (0);
			var y = boxes.GetLength (1);
			var z = boxes.GetLength (2);

			float ymin = 0, xmin = 0, ymax = 0, xmax = 0;

			using (var editor = new ImageEditor (inputFile, outputFile)) {
				for (int i = 0; i < x; i++) {
					for (int j = 0; j < y; j++) {
						if (scores [i, j] < minScore) continue;

						for (int k = 0; k < z; k++) {
							var box = boxes [i, j, k];
							switch (k) {
							case 0:
								ymin = box;
								break;
							case 1:
								xmin = box;
								break;
							case 2:
								ymax = box;
								break;
							case 3:
								xmax = box;
								break;
							}

						}

						int value = Convert.ToInt32 (classes [i, j]);
						editor.AddBox (xmin, xmax, ymin, ymax, label[value-1]);
					}
				}
			}
		}

        static int cropSize = 384;
        static TFTensor Image2Tensor(string path)
        {
            //先截取感兴趣区域
            Bitmap img_ = new Bitmap(path);
            Rectangle rect = new Rectangle(121, 105, cropSize, cropSize);
            //Rectangle rect = new Rectangle(120, 50, cropSize, cropSize);
            Bitmap img = img_.Clone(rect, PixelFormat.Format8bppIndexed);
            BitmapData srcData = img.LockBits(new Rectangle(0, 0, cropSize, cropSize), ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);
            System.IntPtr srcPtr = srcData.Scan0;
            byte[] cropArr = new byte[cropSize * cropSize];
            System.Runtime.InteropServices.Marshal.Copy(srcPtr, cropArr, 0, cropSize * cropSize);
            img.UnlockBits(srcData);
            TFShape tfs_ = new TFShape(cropSize, cropSize, 1);
            TFTensor tensor = TFTensor.FromBuffer(tfs_, cropArr, 0, cropSize * cropSize);
            return tensor;
        }

        static TFTensor Image2Tensor2(string path)
        {
            //先截取感兴趣区域
            Bitmap img = new Bitmap(path);
            BitmapData srcData = img.LockBits(new Rectangle(0, 0, 224, 224), ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);
            System.IntPtr srcPtr = srcData.Scan0;
            byte[] cropArr = new byte[224 * 224];
            System.Runtime.InteropServices.Marshal.Copy(srcPtr, cropArr, 0, 224 * 224);
            img.UnlockBits(srcData);
            TFShape tfs_ = new TFShape(224, 224, 1);
            TFTensor tensor = TFTensor.FromBuffer(tfs_, cropArr, 0, 224 * 224);
            return tensor;
        }
    }
}
