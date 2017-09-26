#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point32.h>
#include <color_segmentation/Segment.h>
#include <color_segmentation/SegmentArray.h>




#include <boost/filesystem.hpp>

#include <opencv2/highgui/highgui.hpp>


#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <pwd.h>
#include <sys/types.h>


using namespace cv;
using namespace std;


// Parameters for HSV limits
int iLowH = 0;
int iHighH = 179;

int iLowS = 0;
int iHighS = 255;

int iLowV = 0;
int iHighV = 255;

int workspace_min_x = 0;
int workspace_max_x = 0;

int workspace_min_y = 0;
int workspace_max_y = 0;

int workspace_topleft_y = 0;
int workspace_topleft_x = 0;


// Minimum area Parameter for the found segments
float min_segment_area = 10;

// Max area Parameter for the found segments
float max_segment_area = 10000;

// If you want to shrink or enlarge the workspace manually, use these parameters (qhd)
//int workspace_width_offset =  -20;
//int workspace_height_offset = -90;

// (hd)
int workspace_width_offset =  0;
int workspace_height_offset = 0;

// Initial workspace roi initialized as invalid
Rect workspace_roi = Rect(-2,-2,0,0);

// The name of the camera topic
string camera_topic = "/kinect2/hd/image_color";

// Visualization switch
bool visualize = true;
bool control_off = true;

ros::Publisher segment_pub;


RNG rng(12345);


// Get the home path to save data
string getHomePath()
{
    uid_t uid = getuid();
    struct passwd *pw = getpwuid(uid);

    if (pw == NULL) {
        ROS_ERROR("Failed to get homedir. Cannot save configuration file\n");
        return "";
    }

    // printf("%s\n", pw->pw_dir);
    string str(pw->pw_dir);
    return str;

}

// Threshold the image based on HSV limits
cv::Mat thresholdImage(const cv::Mat& img)
{
    Mat imgThresholded;

    //Threshold the image
    inRange(img, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (removes small holes from the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    return imgThresholded;

}

// Find the contours in the image
vector<vector<Point> > extractContours(const cv::Mat& img)
{
    vector<vector<Point> > contours;

    // Find contours without any hierarchy
    cv::findContours( img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    return contours;
}
// Filter the contours based on the area
vector<vector<Point> > filterContours(const vector<vector<Point> >& contours, boost::shared_ptr<vector<vector<Point> > > remaining_contours)
{

    vector<vector<Point> >hulls;

    for( size_t i = 0; i < contours.size(); i++ )
    {
        double area = fabs(contourArea(contours[i]));

        if(area >= min_segment_area && area <= max_segment_area)
        {
            vector<Point> hull;

            convexHull( Mat(contours[i]), hull, false );

            hulls.push_back(hull);

            remaining_contours->push_back(contours[i]);

        }
    }

    return hulls;
}


// Find the segment centers
vector<Point2i> findSegmentCenters(const vector < vector<Point> >& hulls)
{
    vector<Point2i> mc(hulls.size());

    for( size_t i = 0; i < hulls.size(); i++ )
    {
        Moments moment = moments( hulls[i], false );

        mc[i] = Point2i( round(moment.m10/moment.m00) , round(moment.m01/moment.m00) );

    }

    return mc;

}

// Get the mask for segments
vector<Mat> getSegmentMasks(const Mat& img, const vector < vector<Point> >& contours)
{
    vector<Mat> masks(contours.size());

    for(size_t i =0; i < contours.size() ; i++){

        cv::Mat mask = cv::Mat::zeros(img.size(),CV_8UC1);

        //std::cout<<mask.rows<<std::endl;
        vector<vector<Point> > tmpcontour(1);
        tmpcontour[0] = contours[i];

        //std::cout<<contours[i]<<std::endl;

        drawContours(mask,tmpcontour,0,255,-1);

        masks[i] = mask;
    }

    return masks;
}

// Calculate the color of the segment
int calculateSegmentHueColor(const Mat& hueimage, Point2i segmentCenter)
{
    int count = 1;
    int huesum = 0;

    Vec3b pixelval = hueimage.at<Vec3b>(segmentCenter.y,segmentCenter.x);

    huesum += pixelval[0];


    if(hueimage.rows - segmentCenter.y > 0)
    {

        pixelval = hueimage.at<Vec3b>(segmentCenter.y+1,segmentCenter.x);

        huesum += pixelval[0];

        count +=1;
    }

    if(segmentCenter.y > 0)
    {

        pixelval = hueimage.at<Vec3b>(segmentCenter.y-1,segmentCenter.x);

        huesum += pixelval[0];

        count +=1;
    }


    if(hueimage.cols - segmentCenter.x > 0)
    {

        pixelval = hueimage.at<Vec3b>(segmentCenter.y,segmentCenter.x+1);

        huesum += pixelval[0];
        count +=1;
    }

    if(segmentCenter.x> 0)
    {

        pixelval = hueimage.at<Vec3b>(segmentCenter.y,segmentCenter.x-1);

        huesum += pixelval[0];

        count += 1 ;
    }


    return huesum/count;

}

// Calculates the workspace roi and returns invalid roi if there is an error
Rect calculateWorkspaceROI(int image_width, int image_height)
{
    Rect roi = Rect(-1,-1,0,0);

    if(workspace_max_x > workspace_min_x && workspace_max_y > workspace_min_y)
    {
        int width = workspace_max_x-workspace_min_x+workspace_width_offset;
        int height = workspace_max_y-workspace_min_y+workspace_height_offset;

        ROS_INFO("Calculated roi width and height %d %d",width,height);

        // Check if the calculated limits are correct
        if(width > image_width || width < 0 || height > image_height || height < 0)
        {
            ROS_WARN("Workspace limits are not correct, check the workspace parameters. Abondoning the workspace limitation parameters");
            workspace_max_x = workspace_min_x;
            workspace_max_y = workspace_min_y;



            return roi ;
        }

        roi.x = workspace_min_x;
        roi.y = workspace_min_y;
        roi.width = width;
        roi.height = height;

        //Rect roi = Rect(workspace_min_x, workspace_min_y, width, height);

        //return roi;
    }

    return roi;
}

// Sorts the segments based on the distance to the top left corner (0,0)
bool sortSegment(color_segmentation::Segment seg1, color_segmentation::Segment seg2)
{
    return((seg1.pixelposcenterx+seg1.pixelposcentery) < (seg2.pixelposcenterx+seg2.pixelposcentery));

}

vector <float> calculateSegmentAngles(const vector< vector<Point> >& contours)
{
    vector<float> res(contours.size());
    for(size_t i = 0; i < contours.size(); i++)
    {
        // fit bounding rectangle around contour
        cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);

        // read points and angle
        cv::Point2f rect_points[4];
        rotatedRect.points( rect_points );

        cv::Point2f edge1 = cv::Vec2f(rect_points[1].x, rect_points[1].y) - cv::Vec2f(rect_points[0].x, rect_points[0].y);
        cv::Point2f edge2 = cv::Vec2f(rect_points[2].x, rect_points[2].y) - cv::Vec2f(rect_points[1].x, rect_points[1].y);

        cv::Point2f usedEdge = edge1;
        if(cv::norm(edge2) > cv::norm(edge1))
            usedEdge = edge2;

        cv::Point2f reference = cv::Vec2f(1,0); // horizontal edge


        double angle = 180.0f/CV_PI * acos((reference.x*usedEdge.x + reference.y*usedEdge.y) / (cv::norm(reference) *cv::norm(usedEdge)));



        res[i] = angle; // angle
    }


    return res;
}

// Create the segment objects
vector<color_segmentation::Segment> createSegments(const Mat& rgbimage,const Mat& hueimage, const vector< vector<Point> >& hulls, const vector< vector<Point> >& contours)
{
    vector<color_segmentation::Segment> result(hulls.size());

    vector<Point2i> centers = findSegmentCenters(hulls);

    vector<float> angles = calculateSegmentAngles(contours);

    vector<Mat> masks = getSegmentMasks(rgbimage,contours);

    // Create the segments using hull information together with with workspace offset
    for (size_t i =0; i < hulls.size(); i++)
    {
        color_segmentation::Segment asegment;

        asegment.pixelposcenterx = centers[i].x + workspace_min_x;
        asegment.pixelposcentery = centers[i].y + workspace_min_y;

        std::vector<geometry_msgs::Point32> points(hulls[i].size());

        for(size_t j = 0 ; j < hulls[i].size() ; j++ )
        {
            points[j].x = hulls[i][j].x+ workspace_min_x;
            points[j].y = hulls[i][j].y+ workspace_min_y;

            Vec3b pixelval = hueimage.at<Vec3b>(hulls[i][j].y,hulls[i][j].x);

        }


        asegment.pixelhull = points;

        asegment.angle = angles[i];


        /***** Create mask image of the segment ************/

        cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

        Mat Points;

        findNonZero(masks[i],Points);

        Rect min_rect=boundingRect(Points);

        cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;

        cv_ptr->image = rgbimage(min_rect);

        cv_ptr->toImageMsg(asegment.image );

        /******************************************************/

        asegment.averagehue = calculateSegmentHueColor(hueimage,centers[i]);

        // std::cout<<(int)asegment.averagehue<<std::endl;

        result[i] = asegment;

    }



    return result;

}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    Mat imgHSV;
    try
    {
        //cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);

        cvtColor(cv_bridge::toCvShare(msg, "bgr8")->image, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        return;
    }

    Mat imgHSVworkspace = imgHSV;

    // This is the first time we are checking for roi
    if(workspace_roi.x == -2)
    {
        workspace_roi = calculateWorkspaceROI(imgHSV.cols,imgHSV.rows);

    }

    // If ROI is correctly calculated
    if(workspace_roi.x >0)
        imgHSVworkspace = imgHSV(workspace_roi);
    else
    {
        workspace_max_y = imgHSV.rows;
        workspace_max_x = imgHSV.cols;

        workspace_min_x = 0;
        workspace_min_y = 0;
    }

    cv::Mat thresholdedImage = thresholdImage(imgHSVworkspace);

    if(!control_off)
        imshow("Thresholded Image", thresholdedImage); //show the thresholded image

    vector<vector<Point> > contours = extractContours(thresholdedImage);

    boost::shared_ptr< vector < vector<Point> > > remaining_contours(new vector< vector<Point> >);

    vector<vector<Point> > hulls = filterContours(contours, remaining_contours);

    Mat imgBGRworkspace;

    cvtColor(imgHSVworkspace,imgBGRworkspace,COLOR_HSV2BGR);

    vector<color_segmentation::Segment> segments = createSegments(imgBGRworkspace,imgHSVworkspace,hulls,*remaining_contours);

    std::sort (segments.begin(), segments.end(), sortSegment);

    color_segmentation::SegmentArray array;

    array.segments = segments;



    if(visualize)
    {

        for( int i = 0; i< segments.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            //  drawContours(imgBGRworkspace , hulls, i, color, 2, 8, NULL, 0, Point() );
            Point2i segmentcenter;
            segmentcenter.x = segments[i].pixelposcenterx-workspace_min_x;
            segmentcenter.y = segments[i].pixelposcentery-workspace_min_y;

            circle(imgBGRworkspace,segmentcenter,2,color);
            stringstream ss;
            ss<<i;
            //std::cout<<i<<std::endl;
            cv::putText(imgBGRworkspace,ss.str(),segmentcenter,0,2,color);
        }


        imshow( "Segments", imgBGRworkspace );


    }

    segment_pub.publish(array);

    if(!control_off || visualize)
        cv::waitKey(30);

}


void saveConfig(int state, void* userdata)
{


    string configpath = getHomePath();

    configpath += "/.ros/color_segmentation/";

    boost::filesystem::path dir(configpath);

    if(!(boost::filesystem::exists(dir)))
    {
        std::cout<<"Doesn't Exists"<<std::endl;
    }

    if (boost::filesystem::create_directory(dir))
        std::cout << "....Successfully Created !" << std::endl;

    configpath += "param.txt";

    ofstream stream(configpath.data());



    if(stream.is_open()){

        stream<<iLowH<<"\n"<<iHighH<<"\n"<<iLowS<<"\n"<<iHighS<<"\n"<<iLowV<<"\n"<<iHighV;

        stream.close();

        ROS_INFO("Parameter Configuration Successfully Saved!");

        destroyWindow("Control");

    }
    else
    {
        ROS_ERROR("Configuration File cannot be opened!!");

    }



}

bool readColorConfig()
{
    string configpath = getHomePath();

    configpath += "/.ros/color_segmentation/";

    configpath += "param.txt";

    ifstream stream(configpath.data());

    if(stream.is_open())
    {
        string str;
        int count = 0;
        while(getline(stream, str))
        {

            std::istringstream ss(str);

            // std::cout<<str<<endl;

            switch(count)
            {
            case 0:
                iLowH = atoi(str.data());
            case 1:
                iHighH = atoi(str.data());
            case 2:
                iLowS  = atoi(str.data());
            case 3:
                iHighS = atoi(str.data());
            case 4:
                iLowV = atoi(str.data());
            case 5:
                iHighV = atoi(str.data());

            }

            count++;

        }


        stream.close();

    }
    else
    {
        return false;
    }

    return true;
}
bool readWorkspaceConfig(int *minX, int *maxX, int *minY, int *maxY,int *topLeftX,int *topLeftY)
{
    string configpath = getHomePath();

    configpath += "/.ros/workspace_segmentation/";

    configpath += "workspace.txt";

    ifstream stream(configpath.data());

    if(stream.is_open())
    {
        string str;
        int count = 0;
        while(getline(stream, str))
        {

            std::istringstream ss(str);

            //std::cout<<str<<endl;

            switch(count)
            {
            case 0:
                *topLeftX = atoi(str.data());
            case 1:
                *topLeftY = atoi(str.data());
            case 2:
                *minX = atoi(str.data());
            case 3:
                *maxX = atoi(str.data());
            case 4:
                *minY  = atoi(str.data());
            case 5:
                *maxY = atoi(str.data());
            default:
                break;

            }

            count++;

        }


        stream.close();



    }
    else
    {
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "color_segmentation_node");
    ros::NodeHandle nh;
    // Private node handle
    ros::NodeHandle pnh("~");

    pnh.getParam("control_off",control_off);
    pnh.getParam("min_segment_area",min_segment_area);
    pnh.getParam("max_segment_area",max_segment_area);
    pnh.getParam("camera_topic",camera_topic);
    pnh.getParam("workspace_width_offset",workspace_width_offset);
    pnh.getParam("workspace_height_offset",workspace_height_offset);
    pnh.getParam("visualize",visualize);

    // If we cannot read the color config or we need the control screen to be shown
    if(!readColorConfig() || !control_off)
    {
        ROS_WARN("Could not read color configuration! Activating control panel");

        //cv::namedWindow("view");

        namedWindow("Control",CV_WINDOW_AUTOSIZE ); //create a window called "Control"
        string name = "Save Config";

        //Create trackbars in "Control" window
        cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
        cvCreateTrackbar("HighH", "Control", &iHighH, 179);

        cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
        cvCreateTrackbar("HighS", "Control", &iHighS, 255);

        cvCreateTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
        cvCreateTrackbar("HighV", "Control", &iHighV, 255);

        cvCreateButton(name.data(),saveConfig);


    }

    if(visualize)
    {
        /// Show in a window
        namedWindow( "Segments", CV_WINDOW_AUTOSIZE );
    }
    cv::startWindowThread();

    if(!readWorkspaceConfig(&workspace_min_x,&workspace_max_x,&workspace_min_y,&workspace_max_y,&workspace_topleft_x,&workspace_topleft_y))
    {
        ROS_WARN("Could not read workspace dimensions! Working on whole image");
    }

    ROS_INFO("Workspace parameters, x limits: %d %d y limits: %d %d offsets: %d %d",workspace_min_x, workspace_max_x, workspace_min_y, workspace_max_y, workspace_width_offset, workspace_height_offset);


    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe(camera_topic, 1, imageCallback);

    segment_pub = nh.advertise<color_segmentation::SegmentArray>("color_segmentation/segments",1);

    ros::spin();
    cv::destroyAllWindows();
}


