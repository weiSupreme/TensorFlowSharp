using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace ExampleCommon
{
	/// <summary>
	/// Allows to add graphic elements to the existing image.
	/// </summary>
	public class ImageEditor : IDisposable
	{
		private Graphics _graphics;
		private Image _image;
		private string _fontFamily;
		private float _fontSize;
		private string _outputFile;
        private Bitmap bmp;

		public ImageEditor (string inputFile, string outputFile, string fontFamily = "Ariel", float fontSize = 12)
		{
			if (string.IsNullOrEmpty (inputFile)) {
				throw new ArgumentNullException (nameof (inputFile));
			}

			if (string.IsNullOrEmpty (outputFile)) {
				throw new ArgumentNullException (nameof (outputFile));
			}

			_fontFamily = fontFamily;
			_fontSize = fontSize;
			_outputFile = outputFile;

			_image = Bitmap.FromFile(inputFile);
            bmp = new Bitmap(_image.Width, _image.Height, PixelFormat.Format32bppArgb);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                g.DrawImage(_image, 0, 0);
            }
            _graphics = Graphics.FromImage (bmp);
		}

		/// <summary>
		/// Adds rectangle with a label in particular position of the image
		/// </summary>
		/// <param name="xmin"></param>
		/// <param name="xmax"></param>
		/// <param name="ymin"></param>
		/// <param name="ymax"></param>
		/// <param name="text"></param>
		/// <param name="colorName"></param>
		public void AddBox (float xmin, float xmax, float ymin, float ymax, string text = "", string colorName = "red")
		{
            var left = xmin * 384 + 121;// _image.Width;
            var right = xmax * 384 + 121;// _image.Width;
            var top = ymin * 384 + 105;// _image.Height;
            var bottom = ymax * 384 + 105;// _image.Height;


			var imageRectangle = new Rectangle (new Point (0, 0), new Size (_image.Width, _image.Height));
			_graphics.DrawImage (bmp, imageRectangle);

			Color color = Color.FromName(colorName);
			Brush brush = new SolidBrush (color);
			Pen pen = new Pen (brush);

			_graphics.DrawRectangle (pen, left, top, right - left, bottom - top);
			var font = new Font (_fontFamily, _fontSize);
			SizeF size = _graphics.MeasureString (text, font);
			_graphics.DrawString (text, font, brush, new PointF (left, top - size.Height));
		}

		public void Dispose ()
		{
			if (bmp != null) {
				bmp.Save (_outputFile);

				if (_graphics != null) {
					_graphics.Dispose ();
				}

				_image.Dispose ();
			}
		}
	}
}
