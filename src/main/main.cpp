#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/aruco.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;
using namespace aruco;

int main(int argc, char *argv[])
{
    // Подключаем словарь ArUco-маркеров
    const auto dictionary = getPredefinedDictionary(DICT_6X6_250);

    // Создаём таймер
    auto timer = TickMeter();

    // Кодек
    auto fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');

    // Захватываем изображение с веб-камеры
    auto webcam = VideoCapture();
    webcam.open(0, CAP_DSHOW);
    
    webcam.set(CAP_PROP_FRAME_WIDTH, 640);
    webcam.set(CAP_PROP_FRAME_HEIGHT, 480);
    webcam.set(CAP_PROP_FOURCC, fourcc);

    // Захватываем изображение из видео
    auto video = VideoCapture("src/videos/Imagine Dragons - Believer.webm");
    auto fps = video.get(CAP_PROP_FPS);

    // Кадры с веб-камеры и из видео
    auto frame = Mat();
    auto image = Mat();

    webcam.read(frame);
    video.read(image);

    // Начинаем отсчёт
    timer.start();

    auto endProgram = false;
    while (endProgram == false)
    {
        // Углы вставки кадра видео
        static vector<Point2f> roiCorners =
        {
            Point2f(5 , 5 ),
            Point2f(45, 5 ),
            Point2f(45, 25),
            Point2f(5 , 25)
        };

        // Матрица гомографии
        static auto warpMat = Mat();

        // Получаем кадр с веб-камеры
        if (webcam.grab() != false)
        {
            webcam.retrieve(frame);
        }

        // Получаем кадр из видео
        timer.stop();
        if (timer.getTimeMilli() >= 1000.0 / fps)
        {
            video.read(image);
            timer.reset();
        }
        cout << timer.getTimeMilli() << endl;
        timer.start();

        // Если видео закончилось, открываем его вновь
        if (image.empty() != false)
        {
            video = VideoCapture("src/videos/Imagine Dragons - Believer.webm");
            continue;
        }

        // Углы кадра из видео
        vector<Point2f> dstCorners;
        dstCorners.emplace_back(Point2f(0, 0));
        dstCorners.emplace_back(Point2f(static_cast<float>(image.cols - 1), 0));
        dstCorners.emplace_back(Point2f(static_cast<float>(image.cols - 1),
                                        static_cast<float>(image.rows - 1)));
        dstCorners.emplace_back(Point2f(0, static_cast<float>(image.rows - 1)));

        // Идентификаторы и углы маркеров
        vector<int> ids;
        vector<vector<Point2f>> corners;

        // Детектируем маркеры
        detectMarkers(frame, dictionary, corners, ids);

        // Если найден хоть один маркер, обновляем координаты углов для вставки
        if (corners.empty() == false)
        {
            // drawDetectedMarkers(frame, corners, ids);
            for (auto i = 0; i < ids.size(); i++)
            {
                switch (ids.at(i))
                {
                    case 2:
                    {
                        roiCorners.at(0) = corners.at(i).at(0);
                        break;
                    }
                    case 28:
                    {
                        roiCorners.at(1) = corners.at(i).at(1);
                        break;
                    }
                    case 20:
                    {
                        roiCorners.at(2) = corners.at(i).at(2);
                        break;
                    }
                    case 16:
                    {
                        roiCorners.at(3) = corners.at(i).at(3);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }
            }
        }

        // Если количество углов кадра и углов для вставки кадра совпадает,
        // определяем матрицу гомографии
        if (dstCorners.size() == roiCorners.size())
        {
            warpMat = getPerspectiveTransform(dstCorners, roiCorners);
        }

        // Выходное изображение и маска
        auto result = frame.clone();
        auto mask = frame.clone();

        // Если матрица ненулевая
        if (warpMat.empty() == false)
        {
            // Накладываем белую маску
            auto whiteImage = Mat(image.size(), image.type(), Scalar(0xFF, 0xFF, 0xFF));
            warpPerspective(whiteImage, mask, warpMat, mask.size());
            result -= mask;

            // Накладываем маску с преобразованным кадром из видео на кадр с веб-камеры
            warpPerspective(image, mask, warpMat, mask.size());
            result += mask;
        }

        // Выводим результат
        imshow("Result", result);

        const auto c = static_cast<char>(waitKey(1));
        if (c == 'q' || c == 'Q' || c == 27)
        {
            endProgram = true;
        }
    }

    return 0;
}
