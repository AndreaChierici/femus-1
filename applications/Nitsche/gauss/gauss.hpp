
const std::vector < std::vector < double > >  tetGauss1 = {{0.16666666666667},
  {0.25},
  {0.25},
  {0.25}
};

const std::vector < std::vector < double > > tetGauss3 = {{ -0.13333333333333, 0.075, 0.075, 0.075, 0.075},
  {0.25, 0.5, 0.16666666666667, 0.16666666666667, 0.16666666666667},
  {0.25, 0.16666666666667, 0.5, 0.16666666666667, 0.16666666666667},
  {0.25, 0.16666666666667, 0.16666666666667, 0.5, 0.16666666666667}
};

const std::vector < std::vector < double > > tetGauss5 = {{0.030283678097089, 0.006026785714286, 0.006026785714286, 0.006026785714286, 0.006026785714286, 0.011645249086029, 0.011645249086029, 0.011645249086029, 0.011645249086029, 0.010949141561386, 0.010949141561386, 0.010949141561386, 0.010949141561386, 0.010949141561386, 0.010949141561386},
  {0.25, 0, 0.33333333333333, 0.33333333333333, 0.33333333333333, 0.72727272727273, 0.090909090909091, 0.090909090909091, 0.090909090909091, 0.43344984642634, 0.43344984642634, 0.43344984642634, 0.066550153573664, 0.066550153573664, 0.066550153573664},
  {0.25, 0.33333333333333, 0, 0.33333333333333, 0.33333333333333, 0.090909090909091, 0.72727272727273, 0.090909090909091, 0.090909090909091, 0.43344984642634, 0.066550153573664, 0.066550153573664, 0.43344984642634, 0.43344984642634, 0.066550153573664},
  {0.25, 0.33333333333333, 0.33333333333333, 0, 0.33333333333333, 0.090909090909091, 0.090909090909091, 0.72727272727273, 0.090909090909091, 0.066550153573664, 0.43344984642634, 0.066550153573664, 0.43344984642634, 0.066550153573664, 0.43344984642634}
};

const std::vector < std::vector < double > > tetGauss7 = {{0.018264223466109, 0.010599941524414, 0.010599941524414, 0.010599941524414, 0.010599941524414, -0.06251774011433, -0.06251774011433, -0.06251774011433, -0.06251774011433, 0.0048914252630735, 0.0048914252630735, 0.0048914252630735, 0.0048914252630735, 0.0009700176366843, 0.0009700176366843, 0.0009700176366843, 0.0009700176366843, 0.0009700176366843, 0.0009700176366843, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985, 0.027557319223985},
  {0.25, 0.76536042300904, 0.078213192330319, 0.078213192330319, 0.078213192330319, 0.63447035000829, 0.1218432166639, 0.1218432166639, 0.1218432166639, 0.0023825066607383, 0.33253916444642, 0.33253916444642, 0.33253916444642, 0, 0.5, 0.5, 0.5, 0, 0, 0.2, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1, 0.2, 0.6, 0.1, 0.2, 0.6},
  {0.25, 0.078213192330319, 0.078213192330319, 0.078213192330319, 0.76536042300904, 0.1218432166639, 0.1218432166639, 0.1218432166639, 0.63447035000829, 0.33253916444642, 0.33253916444642, 0.33253916444642, 0.0023825066607383, 0.5, 0, 0.5, 0, 0.5, 0, 0.1, 0.2, 0.1, 0.1, 0.6, 0.1, 0.2, 0.6, 0.1, 0.6, 0.1, 0.2},
  {0.25, 0.078213192330319, 0.078213192330319, 0.76536042300904, 0.078213192330319, 0.1218432166639, 0.1218432166639, 0.63447035000829, 0.1218432166639, 0.33253916444642, 0.33253916444642, 0.0023825066607383, 0.33253916444642, 0.5, 0.5, 0, 0, 0, 0.5, 0.1, 0.1, 0.2, 0.1, 0.1, 0.6, 0.6, 0.1, 0.2, 0.2, 0.6, 0.1}
};

const std::vector < std::vector < double > > tetGauss8 = {{ -0.039327006641293, 0.0040813160593427, 0.0040813160593427, 0.0040813160593427, 0.0040813160593427, 0.00065808677330435, 0.00065808677330435, 0.00065808677330435, 0.00065808677330435, 0.0043842588251228, 0.0043842588251228, 0.0043842588251228, 0.0043842588251228, 0.0043842588251228, 0.0043842588251228, 0.01383006384251, 0.01383006384251, 0.01383006384251, 0.01383006384251, 0.01383006384251, 0.01383006384251, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0042404374246837, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142, 0.0022387397396142},
  {0.25, 0.61758719030008, 0.12747093656664, 0.12747093656664, 0.12747093656664, 0.9037635088221, 0.032078830392632, 0.032078830392632, 0.032078830392632, 0.45022290435672, 0.049777095643281, 0.049777095643281, 0.049777095643281, 0.45022290435672, 0.45022290435672, 0.31626955260145, 0.18373044739855, 0.18373044739855, 0.18373044739855, 0.31626955260145, 0.31626955260145, 0.022917787844817, 0.23190108939715, 0.23190108939715, 0.51328003336088, 0.23190108939715, 0.23190108939715, 0.23190108939715, 0.022917787844817, 0.51328003336088, 0.23190108939715, 0.022917787844817, 0.51328003336088, 0.73031342780754, 0.037970048471829, 0.037970048471829, 0.1937464752488, 0.037970048471829, 0.037970048471829, 0.037970048471829, 0.73031342780754, 0.1937464752488, 0.037970048471829, 0.73031342780754, 0.1937464752488},
  {0.25, 0.12747093656664, 0.12747093656664, 0.12747093656664, 0.61758719030008, 0.032078830392632, 0.032078830392632, 0.032078830392632, 0.9037635088221, 0.049777095643281, 0.45022290435672, 0.049777095643281, 0.45022290435672, 0.049777095643281, 0.45022290435672, 0.18373044739855, 0.31626955260145, 0.18373044739855, 0.31626955260145, 0.18373044739855, 0.31626955260145, 0.23190108939715, 0.022917787844817, 0.23190108939715, 0.23190108939715, 0.51328003336088, 0.23190108939715, 0.022917787844817, 0.51328003336088, 0.23190108939715, 0.51328003336088, 0.23190108939715, 0.022917787844817, 0.037970048471829, 0.73031342780754, 0.037970048471829, 0.037970048471829, 0.1937464752488, 0.037970048471829, 0.73031342780754, 0.1937464752488, 0.037970048471829, 0.1937464752488, 0.037970048471829, 0.73031342780754},
  {0.25, 0.12747093656664, 0.12747093656664, 0.61758719030008, 0.12747093656664, 0.032078830392632, 0.032078830392632, 0.9037635088221, 0.032078830392632, 0.049777095643281, 0.049777095643281, 0.45022290435672, 0.45022290435672, 0.45022290435672, 0.049777095643281, 0.18373044739855, 0.18373044739855, 0.31626955260145, 0.31626955260145, 0.31626955260145, 0.18373044739855, 0.23190108939715, 0.23190108939715, 0.022917787844817, 0.23190108939715, 0.23190108939715, 0.51328003336088, 0.51328003336088, 0.23190108939715, 0.022917787844817, 0.022917787844817, 0.51328003336088, 0.23190108939715, 0.037970048471829, 0.037970048471829, 0.73031342780754, 0.037970048471829, 0.037970048471829, 0.1937464752488, 0.1937464752488, 0.037970048471829, 0.73031342780754, 0.73031342780754, 0.1937464752488, 0.037970048471829}
};

const std::vector < std::vector < std::vector < double > > > &tetGauss = { tetGauss1, tetGauss1, tetGauss3, tetGauss3, tetGauss5, tetGauss5, tetGauss7, tetGauss7, tetGauss8};



const std::vector < std::vector < double > > triGauss1 = {{0.5},
  {0.33333333333333},
  {0.33333333333333}
};

const std::vector < std::vector < double > > triGauss3 = {{ -0.28125, 0.26041666666667, 0.26041666666667, 0.26041666666667},
  {0.33333333333333, 0.6, 0.2, 0.2},
  {0.33333333333333, 0.2, 0.6, 0.2}
};

const std::vector < std::vector < double > > triGauss5 = {{0.1125, 0.062969590272414, 0.062969590272414, 0.062969590272414, 0.066197076394253, 0.066197076394253, 0.066197076394253},
  {0.33333333333333, 0.79742698535309, 0.10128650732346, 0.10128650732346, 0.05971587178977, 0.47014206410511, 0.47014206410511},
  {0.33333333333333, 0.10128650732346, 0.79742698535309, 0.10128650732346, 0.47014206410511, 0.05971587178977, 0.47014206410511}
};

const std::vector < std::vector < double > > triGauss7 = {{ -0.074785022233835, 0.087807628716602, 0.087807628716602, 0.087807628716602, 0.026673617804419, 0.026673617804419, 0.026673617804419, 0.038556880445128, 0.038556880445128, 0.038556880445128, 0.038556880445128, 0.038556880445128, 0.038556880445128},
  {0.33333333333333, 0.47930806784192, 0.26034596607904, 0.26034596607904, 0.86973979419557, 0.065130102902216, 0.065130102902216, 0.63844418856981, 0.63844418856981, 0.048690315425316, 0.048690315425316, 0.31286549600488, 0.31286549600488},
  {0.33333333333333, 0.26034596607904, 0.47930806784192, 0.26034596607904, 0.065130102902216, 0.86973979419557, 0.065130102902216, 0.048690315425316, 0.31286549600488, 0.63844418856981, 0.31286549600488, 0.63844418856981, 0.048690315425316}
};

const std::vector < std::vector < double > > triGauss9 = {{0.048567898141398, 0.01566735011357, 0.01566735011357, 0.01566735011357, 0.038913770502388, 0.038913770502388, 0.038913770502388, 0.039823869463605, 0.039823869463605, 0.039823869463605, 0.012788837829349, 0.012788837829349, 0.012788837829349, 0.021641769688645, 0.021641769688645, 0.021641769688645, 0.021641769688645, 0.021641769688645, 0.021641769688645},
  {0.33333333333333, 0.020634961602526, 0.48968251919874, 0.48968251919874, 0.12582081701413, 0.43708959149294, 0.43708959149294, 0.62359292876194, 0.18820353561903, 0.18820353561903, 0.91054097321109, 0.044729513394453, 0.044729513394453, 0.7411985987845, 0.7411985987845, 0.036838412054736, 0.036838412054736, 0.22196298916077, 0.22196298916077},
  {0.33333333333333, 0.48968251919874, 0.020634961602526, 0.48968251919874, 0.43708959149294, 0.12582081701413, 0.43708959149294, 0.18820353561903, 0.62359292876194, 0.18820353561903, 0.044729513394453, 0.91054097321109, 0.044729513394453, 0.036838412054736, 0.22196298916077, 0.7411985987845, 0.22196298916077, 0.7411985987845, 0.036838412054736}
};

const std::vector < std::vector < double > > triGauss11 = {{0.043988650581111, 0.0043721557768681, 0.0043721557768681, 0.0043721557768681, 0.019040785996968, 0.019040785996968, 0.019040785996968, 0.0094277240280656, 0.0094277240280656, 0.0094277240280656, 0.03607984877237, 0.03607984877237, 0.03607984877237, 0.034664569352769, 0.034664569352769, 0.034664569352769, 0.020528157714644, 0.020528157714644, 0.020528157714644, 0.020528157714644, 0.020528157714644, 0.020528157714644, 0.0036811918916503, 0.0036811918916503, 0.0036811918916503, 0.0036811918916503, 0.0036811918916503, 0.0036811918916503},
  {0.33333333333333, 0.94802171814342, 0.025989140928288, 0.025989140928288, 0.81142499470415, 0.094287502647923, 0.094287502647923, 0.010726449965571, 0.49463677501721, 0.49463677501721, 0.58531323477097, 0.20734338261451, 0.20734338261451, 0.12218438859902, 0.43890780570049, 0.43890780570049, 0.67793765488259, 0.67793765488259, 0.044841677589131, 0.044841677589131, 0.27722066752828, 0.27722066752828, 0.85887028128264, 0.85887028128264, 0, 0, 0.14112971871736, 0.14112971871736},
  {0.33333333333333, 0.025989140928288, 0.94802171814342, 0.025989140928288, 0.094287502647923, 0.81142499470415, 0.094287502647923, 0.49463677501721, 0.010726449965571, 0.49463677501721, 0.20734338261451, 0.58531323477097, 0.20734338261451, 0.43890780570049, 0.12218438859902, 0.43890780570049, 0.044841677589131, 0.27722066752828, 0.67793765488259, 0.27722066752828, 0.67793765488259, 0.044841677589131, 0, 0.14112971871736, 0.85887028128264, 0.14112971871736, 0.85887028128264, 0}
};

const std::vector < std::vector < double > > triGauss13 = {{0.025869883032872, 0.0040038997777824, 0.0040038997777824, 0.0040038997777824, 0.023434449490911, 0.023434449490911, 0.023434449490911, 0.023295470091988, 0.023295470091988, 0.023295470091988, 0.015508471656898, 0.015508471656898, 0.015508471656898, 0.0053958063683156, 0.0053958063683156, 0.0053958063683156, 0.016097767121216, 0.016097767121216, 0.016097767121216, 0.0077229171053508, 0.0077229171053508, 0.0077229171053508, 0.0077229171053508, 0.0077229171053508, 0.0077229171053508, 0.0089114949615893, 0.0089114949615893, 0.0089114949615893, 0.0089114949615893, 0.0089114949615893, 0.0089114949615893, 0.018519341840692, 0.018519341840692, 0.018519341840692, 0.018519341840692, 0.018519341840692, 0.018519341840692},
  {0.33333333333333, 0.95027566292411, 0.024862168537947, 0.024862168537947, 0.17161491492384, 0.41419254253808, 0.41419254253808, 0.53941224367719, 0.2302938781614, 0.2302938781614, 0.77216003667653, 0.11391998166173, 0.11391998166173, 0.0090853999498354, 0.49545730002508, 0.49545730002508, 0.062277290305887, 0.46886135484706, 0.46886135484706, 0.022076289653624, 0.022076289653624, 0.85130650417435, 0.85130650417435, 0.12661720617203, 0.12661720617203, 0.018620522802521, 0.018620522802521, 0.68944197072859, 0.68944197072859, 0.29193750646889, 0.29193750646889, 0.096506481292159, 0.096506481292159, 0.63586785943387, 0.63586785943387, 0.26762565927397, 0.26762565927397},
  {0.33333333333333, 0.024862168537947, 0.95027566292411, 0.024862168537947, 0.41419254253808, 0.17161491492384, 0.41419254253808, 0.2302938781614, 0.53941224367719, 0.2302938781614, 0.11391998166173, 0.77216003667653, 0.11391998166173, 0.49545730002508, 0.0090853999498354, 0.49545730002508, 0.46886135484706, 0.062277290305887, 0.46886135484706, 0.85130650417435, 0.12661720617203, 0.022076289653624, 0.12661720617203, 0.022076289653624, 0.85130650417435, 0.68944197072859, 0.29193750646889, 0.018620522802521, 0.29193750646889, 0.018620522802521, 0.68944197072859, 0.63586785943387, 0.26762565927397, 0.096506481292159, 0.26762565927397, 0.096506481292159, 0.63586785943387}
};

const std::vector < std::vector < std::vector < double > > > & triGauss = {triGauss1, triGauss1, triGauss3, triGauss3, triGauss5, triGauss5, triGauss7, triGauss7, triGauss9, triGauss9, triGauss11, triGauss11, triGauss13, triGauss13};




const std::vector < std::vector < double > > lineGauss1 = {{2}, {0}
};

const std::vector < std::vector < double > > lineGauss3 = {{1, 1}, { -0.57735026918963, 0.57735026918963}
};

const std::vector < std::vector < double > > lineGauss5 = {{0.55555555555556, 0.88888888888889, 0.55555555555556}, { -0.77459666924148, 0, 0.77459666924148}
};

const std::vector < std::vector < double > > lineGauss7 = {{0.34785484513745, 0.65214515486255, 0.65214515486255, 0.34785484513745}, { -0.86113631159405, -0.33998104358486, 0.33998104358486, 0.86113631159405}
};

const std::vector < std::vector < double > > lineGauss9 = {{0.23692688505619, 0.47862867049937, 0.56888888888889, 0.47862867049937, 0.23692688505619}, { -0.90617984593866, -0.53846931010568, 0, 0.53846931010568, 0.90617984593866}
};

const std::vector < std::vector < double > > lineGauss11 = {{0.3607615730481388, 0.3607615730481388, 0.4679139345726911, 0.4679139345726911, 0.17132449237917097, 0.17132449237917097}, {0.6612093864662645, -0.6612093864662645, -0.23861918608319715, 0.23861918608319715, -0.9324695142031519, 0.9324695142031519}
};

const std::vector < std::vector < double > > lineGauss13 = {{0.417959183673469, 0.381830050505119, 0.279705391489277, 0.129484966168870, 0.381830050505119, 0.279705391489277, 0.129484966168870}, {0.000000000000000, 0.405845151377397, 0.741531185599394, 0.949107912342758, -0.405845151377397, -0.741531185599394, -0.949107912342758}
};

const std::vector < std::vector < double > > lineGauss15 = {{0.362683783378362, 0.313706645877887, 0.222381034453375, 0.101228536290376, 0.362683783378362, 0.313706645877887, 0.222381034453375, 0.101228536290376}, { -0.183434642495650, -0.525532409916329, -0.796666477413627, -0.960289856497536, 0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536}
};

const std::vector < std::vector < double > > lineGauss17 = {{0.330239355001260, 0.312347077040003, 0.260610696402935, 0.180648160694857, 0.081274388361574, 0.312347077040003, 0.260610696402935, 0.180648160694857, 0.081274388361574}, {0.000000000000000, -0.324253423403809, -0.613371432700590, -0.836031107326636, -0.968160239507626, 0.324253423403809, 0.613371432700590, 0.836031107326636, 0.968160239507626}
};

const std::vector < std::vector < double > > lineGauss19 = {{0.295524224714753, 0.269266719309996, 0.219086362515982, 0.149451349150581, 0.066671344308688, 0.295524224714753, 0.269266719309996, 0.219086362515982, 0.149451349150581, 0.066671344308688}, { -0.148874338981631, -0.433395394129247, -0.679409568299024, -0.865063366688985, -0.973906528517172, 0.148874338981631, 0.433395394129247, 0.679409568299024, 0.865063366688985, 0.973906528517172}
};

const std::vector < std::vector < double > > lineGauss21 = {{0.272925086777901, 0.262804544510247, 0.233193764591990, 0.186290210927734, 0.125580369464905, 0.055668567116174, 0.262804544510247, 0.233193764591990, 0.186290210927734, 0.125580369464905, 0.055668567116174}, {0.000000000000000, -0.269543155952345, -0.519096129206812, -0.730152005574049, -0.887062599768095, -0.978228658146057, 0.269543155952345, 0.519096129206812, 0.730152005574049, 0.887062599768095, 0.978228658146057}
};

const std::vector < std::vector < double > > lineGauss23 = {{0.249147045813403, 0.233492536538355, 0.203167426723066, 0.160078328543346, 0.106939325995318, 0.047175336386512, 0.249147045813403, 0.233492536538355, 0.203167426723066, 0.160078328543346, 0.106939325995318, 0.047175336386512}, {0.125233408511469, 0.367831498998180, 0.587317954286617, 0.769902674194305, 0.904117256370475, 0.981560634246719, -0.125233408511469, -0.367831498998180, -0.587317954286617, -0.769902674194305, -0.904117256370475, -0.981560634246719}
};

const std::vector < std::vector < double > > lineGauss25 = {{0.232551553230874, 0.226283180262897, 0.207816047536889, 0.178145980761946, 0.138873510219787, 0.092121499837728, 0.040484004765316, 0.226283180262897, 0.207816047536889, 0.178145980761946, 0.138873510219787, 0.092121499837728, 0.040484004765316}, {0.000000000000000, 0.230458315955135, 0.448492751036447, 0.642349339440340, 0.801578090733310, 0.917598399222978, 0.984183054718588, -0.230458315955135, -0.448492751036447, -0.642349339440340, -0.801578090733310, -0.917598399222978, -0.984183054718588}
};

const std::vector < std::vector < double > > lineGauss27 = {{0.215263853463158, 0.205198463721296, 0.185538397477938, 0.157203167158194, 0.121518570687903, 0.080158087159760, 0.035119460331752, 0.215263853463158, 0.205198463721296, 0.185538397477938, 0.157203167158194, 0.121518570687903, 0.080158087159760, 0.035119460331752}, {0.108054948707344, 0.319112368927890, 0.515248636358154, 0.687292904811685, 0.827201315069765, 0.928434883663574, 0.986283808696812, -0.108054948707344, -0.319112368927890, -0.515248636358154, -0.687292904811685, -0.827201315069765, -0.928434883663574, -0.986283808696812}
};

const std::vector < std::vector < double > > lineGauss29 = {{0.202578241925561, 0.198431485327112, 0.186161000015562, 0.166269205816994, 0.139570677926154, 0.107159220467172, 0.070366047488108, 0.030753241996117, 0.198431485327112, 0.186161000015562, 0.166269205816994, 0.139570677926154, 0.107159220467172, 0.070366047488108, 0.030753241996117}, {0.000000000000000, 0.201194093997435, 0.394151347077563, 0.570972172608539, 0.724417731360170, 0.848206583410427, 0.937273392400706, 0.987992518020485, -0.201194093997435, -0.394151347077563, -0.570972172608539, -0.724417731360170, -0.848206583410427, -0.937273392400706, -0.987992518020485}
};

const std::vector < std::vector < double > > lineGauss31 = {{0.189450610455068, 0.182603415044922, 0.169156519395003, 0.149595988816577, 0.124628971255534, 0.095158511682492, 0.062253523938648, 0.027152459411754, 0.189450610455068, 0.182603415044922, 0.169156519395003, 0.149595988816577, 0.124628971255534, 0.095158511682492, 0.062253523938648, 0.027152459411754}, {0.095012509837637, 0.281603550779259, 0.458016777657227, 0.617876244402644, 0.755404408355003, 0.865631202387832, 0.944575023073233, 0.989400934991650, -0.095012509837637, -0.281603550779259, -0.458016777657227, -0.617876244402644, -0.755404408355003, -0.865631202387832, -0.944575023073233, -0.989400934991650}
};

const std::vector < std::vector < double > > lineGauss33 = {{0.179446470356207, 0.176562705366993, 0.168004102156450, 0.154045761076810, 0.135136368468525, 0.111883847193404, 0.085036148317179, 0.055459529373987, 0.024148302868548, 0.176562705366993, 0.168004102156450, 0.154045761076810, 0.135136368468525, 0.111883847193404, 0.085036148317179, 0.055459529373987, 0.024148302868548}, {0.000000000000000, 0.178484181495848, 0.351231763453876, 0.512690537086477, 0.657671159216691, 0.781514003896801, 0.880239153726986, 0.950675521768768, 0.990575475314417, -0.178484181495848, -0.351231763453876, -0.512690537086477, -0.657671159216691, -0.781514003896801, -0.880239153726986, -0.950675521768768, -0.990575475314417}
};

const std::vector < std::vector < double > > lineGauss35 = {{0.169142382963144, 0.164276483745833, 0.154684675126265, 0.140642914670651, 0.122555206711479, 0.100942044106287, 0.076425730254889, 0.049714548894970, 0.021616013526483, 0.169142382963144, 0.164276483745833, 0.154684675126265, 0.140642914670651, 0.122555206711479, 0.100942044106287, 0.076425730254889, 0.049714548894970, 0.021616013526483}, { -0.084775013041735, -0.251886225691505, -0.411751161462843, -0.559770831073948, -0.691687043060353, -0.803704958972523, -0.892602466497556, -0.955823949571398, -0.991565168420931, 0.084775013041735, 0.251886225691505, 0.411751161462843, 0.559770831073948, 0.691687043060353, 0.803704958972523, 0.892602466497556, 0.955823949571398, 0.991565168420931}
};

const std::vector < std::vector < double > > lineGauss37 = {{0.161054449848784, 0.158968843393954, 0.152766042065860, 0.142606702173607, 0.128753962539336, 0.111566645547334, 0.091490021622450, 0.069044542737641, 0.044814226765700, 0.019461788229727, 0.158968843393954, 0.152766042065860, 0.142606702173607, 0.128753962539336, 0.111566645547334, 0.091490021622450, 0.069044542737641, 0.044814226765700, 0.019461788229727}, {0.000000000000000, -0.160358645640225, -0.316564099963630, -0.464570741375961, -0.600545304661681, -0.720966177335229, -0.822714656537143, -0.903155903614818, -0.960208152134830, -0.992406843843584, 0.160358645640225, 0.316564099963630, 0.464570741375961, 0.600545304661681, 0.720966177335229, 0.822714656537143, 0.903155903614818, 0.960208152134830, 0.992406843843584}
};

const std::vector < std::vector < double > > lineGauss39 = {{0.152753387130726, 0.149172986472604, 0.142096109318382, 0.131688638449177, 0.118194531961518, 0.101930119817240, 0.083276741576705, 0.062672048334109, 0.040601429800387, 0.017614007139152, 0.152753387130726, 0.149172986472604, 0.142096109318382, 0.131688638449177, 0.118194531961518, 0.101930119817240, 0.083276741576705, 0.062672048334109, 0.040601429800387, 0.017614007139152}, {0.076526521133497, 0.227785851141645, 0.373706088715420, 0.510867001950827, 0.636053680726515, 0.746331906460151, 0.839116971822219, 0.912234428251326, 0.963971927277914, 0.993128599185095, -0.076526521133497, -0.227785851141645, -0.373706088715420, -0.510867001950827, -0.636053680726515, -0.746331906460151, -0.839116971822219, -0.912234428251326, -0.963971927277914, -0.993128599185095}
};

const std::vector < std::vector < std::vector < double > > > & lineGauss = {
  lineGauss1, lineGauss1, lineGauss3, lineGauss3, lineGauss5, lineGauss5, lineGauss7, lineGauss7,
  lineGauss9, lineGauss9, lineGauss11, lineGauss11, lineGauss13, lineGauss13, lineGauss15, lineGauss15,
  lineGauss17, lineGauss17,  lineGauss19, lineGauss19, lineGauss21, lineGauss21, lineGauss23, lineGauss23,
  lineGauss25, lineGauss25, lineGauss27, lineGauss27, lineGauss29, lineGauss29,lineGauss31, lineGauss31, 
  lineGauss33, lineGauss33, lineGauss35, lineGauss35, lineGauss37, lineGauss37, lineGauss39, lineGauss39
};
