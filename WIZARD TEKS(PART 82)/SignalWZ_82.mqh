//+------------------------------------------------------------------+
//|                                  SignalRL_Ichimoku_ADXWilder.mqh |
//|                             Copyright 2000-2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>

#resource "Python/82_1_0.onnx" as uchar __82_1[]
#resource "Python/82_4_0.onnx" as uchar __82_4[]
#resource "Python/82_5_0.onnx" as uchar __82_5[]
#include <SRI\PipeLine.mqh>
#define __PATTERNS 3
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Signals of Reinforcement Learning with TRIX and Williams Percent Range  |
//| Type=SignalAdvanced                                              |
//| Name=TRIX and Williams Percent Range                             |
//| ShortName=RL_TRIX_WPR                                            |
//| Class=CSignalRL_TRX_WPR                                          |
//| Page=signal_trix_wpr                                             |
//| Parameter=Pattern_1,int,50,Pattern 1                             |
//| Parameter=Pattern_4,int,50,Pattern 4                             |
//| Parameter=Pattern_5,int,50,Pattern 5                             |
//| Parameter=Look_Back,int,30,Scaler Look Back Period               |
//| Parameter=Scaler_Type,int,1,Scaler Type 0-MinMax, 1-Standard, 2-Robust |
//| Parameter=PatternsUsed,int,255,Patterns Used BitMap              |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Class CSignalRL_TRX_WPR.                              |
//| Purpose: Class of generator of trade signals based on            |
//|          Signals of RL with TRIX and Williams Percent Range                |
//| Is derived from the CExpertSignal class.                         |
//+------------------------------------------------------------------+
#define __PERIOD 3
class CSignalRL_TRX_WPR : public CExpertSignal
{
protected:
   CiTriX            m_trix;
   CiWPR             m_wpr;

   int               m_patterns_used;

   long              m_handles[__PATTERNS];
   //--- adjusted parameters

   //--- "weights" of market models (0-100)
   int               m_pattern_1;      // model 1
   int               m_pattern_4;      // model 4
   int               m_pattern_5;      // model 5
   //
   int               m_look_back;
   int               m_scaler_type;
   CPreprocessingPipeline m_pipeline;

public:
   CSignalRL_TRX_WPR(void);
   ~CSignalRL_TRX_WPR(void);
   //--- methods of setting adjustable parameters
   //--- methods of adjusting "weights" of market models
   void              Pattern_1(int value)
   {  m_pattern_1 = value;
   }
   void              Pattern_4(int value)
   {  m_pattern_4 = value;
   }
   void              Pattern_5(int value)
   {  m_pattern_5 = value;
   }
   //
   void              Look_Back(int value)
   {  m_look_back = value;
   }
   void              Scaler_Type(int value)
   {  m_scaler_type = value;
   }
   //
   void              PatternsUsed(int value)
   {  m_patterns_used = value;
      PatternsUsage(value);
   }
   //--- method of verification of settings
   virtual bool      ValidationSettings(void);
   //--- method of creating the oscillator and timeseries
   virtual bool      InitIndicators(CIndicators *indicators);
   //--- methods of checking if the market models are formed
   virtual int       LongCondition(void) override;
   virtual int       ShortCondition(void) override;
   //virtual double    Direction(void) override;

protected:
   //--- method of initialization of the oscillator
   bool              InitRL_TRX_WPR(CIndicators *indicators);
   //--- methods of getting data
   double            TRX(int ind)
   {  //
      m_trix.Refresh(-1);
      return(m_trix.Main(ind));
   }
   double            TRX_MAX(int ind)
   {  //
      m_trix.Refresh(-1);
      return(m_trix.Main(m_trix.Maximum(0, ind, __PERIOD)));
   }
   double            TRX_MIN(int ind)
   {  //
      m_trix.Refresh(-1);
      return(m_trix.Main(m_trix.Minimum(0, ind, __PERIOD)));
   }
   double            WPR(int ind)
   {  //
      m_wpr.Refresh(-1);
      return(m_wpr.Main(ind));
   }
   double            Close(int ind)
   {  //
      m_close.Refresh(-1);
      return(m_close.GetData(ind));
   }
   double            High(int ind)
   {  //
      m_high.Refresh(-1);
      return(m_high.GetData(ind));
   }
   double            Low(int ind)
   {  //
      m_low.Refresh(-1);
      return(m_low.GetData(ind));
   }
   long              Volume(int ind)
   {  //
      m_tick_volume.Refresh(-1);
      return(m_tick_volume.GetData(ind));
   }
   int               X()
   {  //
      return(StartIndex());
   }
   //--- methods to check for patterns
   bool              IsPattern_1(ENUM_POSITION_TYPE T);
   bool              IsPattern_4(ENUM_POSITION_TYPE T);
   bool              IsPattern_5(ENUM_POSITION_TYPE T);

   double            RunModel(int Index, ENUM_POSITION_TYPE T, vectorf &X);
};
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSignalRL_TRX_WPR::CSignalRL_TRX_WPR(void) : m_pattern_4(50),
   m_pattern_1(50),
   m_pattern_5(50)
//m_patterns_usage(255)
{
//--- initialization of protected data
   m_used_series = USE_SERIES_OPEN + USE_SERIES_HIGH + USE_SERIES_LOW + USE_SERIES_CLOSE + USE_SERIES_TICK_VOLUME;
   PatternsUsage(m_patterns_usage);
//--- create model from static buffer
   m_handles[0] = OnnxCreateFromBuffer(__82_1, ONNX_DEFAULT);
   m_handles[1] = OnnxCreateFromBuffer(__82_4, ONNX_DEFAULT);
   m_handles[2] = OnnxCreateFromBuffer(__82_5, ONNX_DEFAULT);
   // 4) Scaling (choose one; scaling applies to all numeric columns now)
   if(m_scaler_type == 0) m_pipeline.AddMinMaxScaler(0.0, 1.0);
   else if(m_scaler_type == 1) m_pipeline.AddStandardScaler();
   else if(m_scaler_type == 2) m_pipeline.AddRobustScaler();
}
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CSignalRL_TRX_WPR::~CSignalRL_TRX_WPR(void)
{
}
//+------------------------------------------------------------------+
//| Validation settings protected data.                              |
//+------------------------------------------------------------------+
bool CSignalRL_TRX_WPR::ValidationSettings(void)
{
//--- validation settings of additional filters
   if(!CExpertSignal::ValidationSettings())
      return(false);
//--- initial data checks
   const ulong _in_shape[] = {1, 2};
   const ulong _out_shape[] = {1, 3};
   for(int i = 0; i < __PATTERNS; i++)
   {  // Set input shapes
      if(!OnnxSetInputShape(m_handles[i], ONNX_DEFAULT, _in_shape))
      {  Print("OnnxSetInputShape error ", GetLastError(), " for feature: ", i);
         return(false);
      }
      // Set output shapes
      if(!OnnxSetOutputShape(m_handles[i], 0, _out_shape))
      {  Print("OnnxSetOutputShape error ", GetLastError(), " for feature: ", i);
         return(false);
      }
   }
//--- ok
   return(true);
}
//+------------------------------------------------------------------+
//| Create indicators.                                               |
//+------------------------------------------------------------------+
bool CSignalRL_TRX_WPR::InitIndicators(CIndicators *indicators)
{
//--- check pointer
   if(indicators == NULL)
      return(false);
//--- initialization of indicators and timeseries of additional filters
   if(!CExpertSignal::InitIndicators(indicators))
      return(false);
//--- create and initialize MA oscillator
   if(!InitRL_TRX_WPR(indicators))
      return(false);
//--- ok
   return(true);
}
//+------------------------------------------------------------------+
//| Initialize MA indicators.                                        |
//+------------------------------------------------------------------+
bool CSignalRL_TRX_WPR::InitRL_TRX_WPR(CIndicators *indicators)
{
//--- check pointer
   if(indicators == NULL)
      return(false);
//--- add object to collection
//--- add object to collection
   if(!indicators.Add(GetPointer(m_trix)))
   {  printf(__FUNCTION__ + ": error adding object");
      return(false);
   }
   if(!indicators.Add(GetPointer(m_wpr)))
   {  printf(__FUNCTION__ + ": error adding object");
      return(false);
   }
//--- initialize object
   if(!m_trix.Create(m_symbol.Name(), m_period, __PERIOD, PRICE_CLOSE))
   {  printf(__FUNCTION__ + ": error initializing object");
      return(false);
   }
   if(!m_wpr.Create(m_symbol.Name(), m_period, __PERIOD))
   {  printf(__FUNCTION__ + ": error initializing object");
      return(false);
   }
//--- ok
   return(true);
}
//+------------------------------------------------------------------+
//| Detecting the "weighted" direction                               |
//+------------------------------------------------------------------+
//double CSignalRL_TRX_WPR::Direction(void)
//{  //return(LongCondition() - ShortCondition());
//}
//+------------------------------------------------------------------+
//| "Voting" that price will grow.                                   |
//+------------------------------------------------------------------+
int CSignalRL_TRX_WPR::LongCondition(void)
{  int result  = 0, results = 0;
   vectorf _x;
   _x.Init(2);
   _x.Fill(0.0);
//--- if the model 1 is used
   if(((m_patterns_usage & 0x02) != 0) && IsPattern_1(POSITION_TYPE_BUY))
   {  _x[0] = 1.0f;
      double _y = RunModel(0, POSITION_TYPE_BUY, _x);
      if(_y > 0.0)
      {  result += m_pattern_1;
         results++;
      }
   }
//--- if the model 4 is used
   if(((m_patterns_usage & 0x10) != 0) && IsPattern_4(POSITION_TYPE_BUY))
   {  _x[0] = 1.0f;
      double _y = RunModel(1, POSITION_TYPE_BUY, _x);
      if(_y > 0.0)
      {  result += m_pattern_4;
         results++;
      }
   }
//--- if the model 5 is used
   if(((m_patterns_usage & 0x20) != 0) && IsPattern_5(POSITION_TYPE_BUY))
   {  _x[0] = 1.0f;
      double _y = RunModel(2, POSITION_TYPE_BUY, _x);
      if(_y > 0.0)
      {  result += m_pattern_5;
         results++;
      }
   }
//--- return the result
//if(result > 0)printf(__FUNCSIG__+" result is: %i",result);
   if(results > 0 && result > 0)
   {  return(int(round(result / results)));
   }
   return(0);
}
//+------------------------------------------------------------------+
//| "Voting" that price will fall.                                   |
//+------------------------------------------------------------------+
int CSignalRL_TRX_WPR::ShortCondition(void)
{  int result  = 0, results = 0;
   vectorf _x;
   _x.Init(2);
   _x.Fill(0.0);
//--- if the model 1 is used
   if(((m_patterns_usage & 0x02) != 0) && IsPattern_1(POSITION_TYPE_SELL))
   {  _x[1] = 1.0f;
      double _y = RunModel(0, POSITION_TYPE_SELL, _x);
      if(_y < 0.0)
      {  result += m_pattern_1;
         results++;
      }
   }
//--- if the model 4 is used
   if(((m_patterns_usage & 0x10) != 0) && IsPattern_4(POSITION_TYPE_SELL))
   {  _x[1] = 1.0f;
      double _y = RunModel(1, POSITION_TYPE_SELL, _x);
      if(_y < 0.0)
      {  result += m_pattern_4;
         results++;
      }
   }
//--- if the model 5 is used
   if(((m_patterns_usage & 0x20) != 0) && IsPattern_5(POSITION_TYPE_SELL))
   {  _x[1] = 1.0f;
      double _y = RunModel(2, POSITION_TYPE_SELL, _x);
      if(_y < 0.0)
      {  result += m_pattern_5;
         results++;
      }
   }
//--- return the result
//if(result > 0)printf(__FUNCSIG__+" result is: %i",result);
   if(results > 0 && result > 0)
   {  return(int(round(result / results)));
   }
   return(0);
}
//+------------------------------------------------------------------+
//| Check for Pattern 1.                                             |
//+------------------------------------------------------------------+
bool CSignalRL_TRX_WPR::IsPattern_1(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY && Low(X() + 1) > Low(X()) && -80.0 > WPR(X()) && TRX(X()) > TRX(X() + 1))
   {  return(true);
   }
   else if(T == POSITION_TYPE_SELL && High(X()) > High(X() + 1) && WPR(X()) > -20.0 && TRX(X() + 1) > TRX(X()))
   {  return(true);
   }
   return(false);
}
//+------------------------------------------------------------------+
//| Check for Pattern 4.                                             |
//+------------------------------------------------------------------+
bool CSignalRL_TRX_WPR::IsPattern_4(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY && 0.0 < TRX(X()) && 0.0 >  TRX(X() + 1) && TRX(X()) == TRX_MAX(X()) && WPR(X()) > -50.0 && WPR(X()) < -20.0)
   {  return(true);
   }
   else if(T == POSITION_TYPE_SELL &&  0.0 < TRX(X() + 1) && 0.0 >  TRX(X()) && TRX(X()) == TRX_MIN(X()) && WPR(X()) > -80.0 && WPR(X()) < -50.0)
   {  return(true);
   }
   return(false);
}
//+------------------------------------------------------------------+
//| Check for Pattern 5.                                             |
//+------------------------------------------------------------------+
bool CSignalRL_TRX_WPR::IsPattern_5(ENUM_POSITION_TYPE T)
{  if(T == POSITION_TYPE_BUY && TRX(X() + 2) < TRX(X() + 1) && TRX(X() + 1) >  TRX(X()) && TRX(X() + 1) == TRX_MAX(X()) && WPR(X() + 1) > -20.0 && WPR(X()) < -20.0)
   {  return(true);
   }
   else if(T == POSITION_TYPE_SELL && TRX(X() + 2) > TRX(X() + 1) && TRX(X() + 1) <  TRX(X()) && TRX(X() + 1) == TRX_MIN(X()) && WPR(X()) > -80.0 && WPR(X() + 1) < -80.0)
   {  return(true);
   }
   return(false);
}
//+------------------------------------------------------------------+
//| Forward Feed Network, to Get Forecast State.                     |
//+------------------------------------------------------------------+
double CSignalRL_TRX_WPR::RunModel(int Index, ENUM_POSITION_TYPE T, vectorf &X)
{  vectorf _y(3);
   _y.Fill(0.0);
   ResetLastError();
   if(!OnnxRun(m_handles[Index], ONNX_NO_CONVERSION, X, _y))
   {  printf(__FUNCSIG__ + " failed to get y forecast, err: %i", GetLastError());
      return(double(_y[0]));
   }
   //printf(__FUNCSIG__ + " y 0: "+DoubleToString(_y[0],5)+ " y 2: "+DoubleToString(_y[2],5));
   float _y_run = 0.0f;
   if(T == POSITION_TYPE_BUY && _y[0] > _y[2] && _y[0] > 0.0)
   {  _y_run = fabs(_y[0])/fabs(_y.Sum());
   }
   else if(T == POSITION_TYPE_SELL && _y[0] < _y[2] && _y[2] < 0.0)
   {  _y_run = fabs(_y[2])/fabs(_y.Sum());
   }
   return(double(_y_run));
}
//+------------------------------------------------------------------+
