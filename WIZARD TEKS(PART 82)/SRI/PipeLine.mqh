//+------------------------------------------------------------------+
//|                                                    PipeLine.mqh  |
//|                   Signal Repository Infrastructure               |
//|                   Preprocessing Pipeline for RL Signals           |
//+------------------------------------------------------------------+
#ifndef SRI_PIPELINE_MQH
#define SRI_PIPELINE_MQH

class CPreprocessingPipeline
{
private:
   int   m_scaler_type;  // 0=MinMax, 1=Standard, 2=Robust
   double m_min_val;
   double m_max_val;

public:
   CPreprocessingPipeline(void) : m_scaler_type(-1), m_min_val(0.0), m_max_val(1.0) {}
   ~CPreprocessingPipeline(void) {}

   void AddMinMaxScaler(double min_val, double max_val)
   {
      m_scaler_type = 0;
      m_min_val = min_val;
      m_max_val = max_val;
   }

   void AddStandardScaler(void)
   {
      m_scaler_type = 1;
   }

   void AddRobustScaler(void)
   {
      m_scaler_type = 2;
   }

   int GetScalerType(void) const { return m_scaler_type; }
};

#endif
