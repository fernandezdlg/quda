/*
  Define functors to allow for generic accessors regardless of field
  ordering.  Currently this is used for cpu fields only with limited
  ordering support, but this will be expanded for device ordering
  also.
 */

template <typename Float>
class ColorSpinorFieldOrder {

 protected:
  cpuColorSpinorField &field;

 public:
  ColorSpinorFieldOrder(cpuColorSpinorField &field) : field(field) { ; }
  virtual ~ColorSpinorFieldOrder() { ; }

  virtual const Float& operator()(const int &x, const int &s, const int &c, const int &z) const = 0;
  virtual Float& operator()(const int &x, const int &s, const int &c, const int &z) = 0;

  int Ncolor() const { return field.Ncolor(); }
  int Nspin() const { return field.Nspin(); }
  int Volume() const { return field.Volume(); }

};

template <typename Float>
class SpaceSpinColorOrder : public ColorSpinorFieldOrder<Float> {

 private:
  cpuColorSpinorField &field; // convenient to have a "local" reference for code brevity

 public:
  SpaceSpinColorOrder(cpuColorSpinorField &field): ColorSpinorFieldOrder<Float>(field), field(field) 
  { ; }
  virtual ~SpaceSpinColorOrder() { ; }

  const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
    unsigned long index = ((x*field.nSpin+s)*field.nColor+c)*2+z;
    return *((Float*)(field.v) + index);
  }

  Float& operator()(const int &x, const int &s, const int &c, const int &z) {
    unsigned long index = ((x*field.nSpin+s)*field.nColor+c)*2+z;
    return *((Float*)(field.v) + index);
  }
};

template <typename Float>
class SpaceColorSpinOrder : public ColorSpinorFieldOrder<Float> {

 private:
  cpuColorSpinorField &field;  // convenient to have a "local" reference for code brevity

 public:
  SpaceColorSpinOrder(cpuColorSpinorField &field) : ColorSpinorFieldOrder<Float>(field), field(field)
  { ; }
  virtual ~SpaceColorSpinOrder() { ; }

  const Float& operator()(const int &x, const int &s, const int &c, const int &z) const {
    unsigned long index = ((x*field.nColor+c)*field.nSpin+s)*2+z;
    return *((Float*)(field.v) + index);
  }

  Float& operator()(const int &x, const int &s, const int &c, const int &z) {
    unsigned long index = ((x*field.nColor+c)*field.nSpin+s)*2+z;    
    return *((Float*)(field.v) + index);
  }
};
