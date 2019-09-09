/*******************************************************************************
Automatic Alignment of Heliostats project

This program extracts heliostat reflection images (HRIs) from a live video feed
of a solar power tower (SPT) central receiver (CR) by using examining the
magnitude of frequency vibration intensity of component pixel vibrations
obtained using a fourier transform of the change of pixel intensity in time.

Project aims at using nVidia 1070Ti (Pascal architecture, GP104) graphics card
to optimise FFT processing. Assumes data lives in a single block of memory but
can be changed in by cuda macros.

This project is maintained on github.

https://github.com/obscuresc/aah-hp

Jack Arney 08-09-19
*******************************************************************************/


#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "src/video_param.h"

#include <cuda_runtime_api.h>
#include <cufft.h>
#include "src/cuda.cu"

// cuda macros
#define RANK          1           //
#define IDIST         1           // distance between 1st elements of batches
#define ISTRIDE       1           // do every ISTRIDEth index
#define ODIST         1           // distance between 1st elements of output
#define OSTRIDE       1           // distance between output elements

// socket macros
#define DOMAIN        AF_INET     // ipv4 (AF_INET) or ipv6 (AF_INET6)
#define PROTOCOL      0           // default
#define TYPE          SOCK_DGRAM  // udp (SOCK_DGRAM) or tcp (SOCK_STREAM)

// server settings
#define BACKQUEUE     5           // max connections in queue
#define PORTNUM       8080

// heliostat deviation parameters @pixelval or physical
struct heliostat_dev_t {

  float x;
  float y;
};


/******************************************************************************/


// sends calculation from PC to remote raspberry pi
bool send_dev_rpi(heliostat_dev_t * heliostat_dev) {

  // create client socket
  int client_socket;
  struct sockaddr_in server_addr;
  struct hostent *server;
  client_socket = socket(DOMAIN, TYPE, PROTOCOL);
  if(client_socket < 0) {
    printf("Communications error: Failed to create socket.\n");
    return -1;
  }

  server_addr.sin_family = DOMAIN;
  server_addr.sin_port = htons(PORTNUM);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  // create packet and send
  const char * message = "@message";
  sendto(client_socket, message, strlen(message), MSG_CONFIRM,
        (const struct sockaddr *) &server_addr, sizeof(server_addr));

  // clean up
  close(client_socket);
  return 0;
}


// perform one dimensional fft
// takes data location, number of elements in dimesions of in and out data
/*
 float * fft1d() {

  size_t BATCH = 1;
  // assemble data
  // cufftReal sample[] = {2.598076211353316, 3.2402830701637395, 3.8494572900049224, 4.419388724529261, 4.944267282795252, 5.41874215947433, 5.837976382931011, 6.197696125093141, 6.494234270429254, 6.724567799874842, 6.886348608602047, 6.97792744346504, 6.998370716093996, 6.9474700202387565, 6.8257442563389, 6.6344343416615565, 6.37549055993378, 6.051552679431957, 5.665923042211819, 5.222532898817316, 4.725902331664744, 4.181094175657916, 3.5936624057845576, 2.9695955178498603, 2.315255479544737, 1.6373128742041732, 0.9426788984240022, 0.23843490677753865, -0.46823977812093664, -1.1701410542749289, -1.8601134815746807, -2.531123226988873, -3.176329770049035, -3.7891556376344524, -4.363353457155562, -4.893069644570959, -5.3729040779788875, -5.797965148448726, -6.163919626883915, -6.467036838555256, -6.704226694973039, -6.873071195387157, -6.971849076777267, -6.999553361041935, -6.955901620504255, -6.84133885708361, -6.657032965782207, -6.404862828733319, -6.0873991611848375, -5.707878304681281, -5.270169234606201, -4.778734118422206, -4.23858282669252, -3.6552218606153755, -3.0345982167228436, -2.383038761007964, -1.707185730522749, -1.0139290199674, -0.31033594356630245, 0.39642081173600463, 1.0991363072871054, 1.7906468025248339, 2.463902784786862, 3.1120408346390414, 3.728453594100783, 4.306857124485735, 4.841354967187034, 5.326498254347925, 5.757341256627454, 6.129491801786784, 6.439156050110601, 6.683177170206378, 6.859067520906216, 6.965034011197066, 6.999996379650895, 6.963598207007518, 6.85621054964381, 6.678928156888352, 6.433558310743566, 6.122602401787424, 5.749230429076629, 5.317248684008804, 4.831060947586139, 4.295623596650021, 3.7163950767501706, 3.0992802567403803, 2.4505702323708074, 1.7768781925409076, 1.0850720020162676, 0.3822041878858906, -0.3245599564963766, -1.0280154171511335, -1.7209909100394047, -2.3964219877733033, -3.0474230571943477, -3.667357573646071, -4.249905696354359, -4.78912871521179, -5.279529592175676, -5.716109000098287};

cufftReal sample[] = {0.0, 0.496882714764706, 0.956769360011055, 1.3458758065781264, 1.6365415651669575, 1.8095699271693415, 1.8557802568827082, 1.7766319173099563, 1.5838684495799855, 1.2982244548656607, 0.9473270250923755, 0.5629998564463234, 0.17823397156835918, -0.17588110061848172, -0.47297062059564154, -0.6936648548689307, -0.827152115976356, -0.8718546105842898, -0.8351744585913987, -0.7323493025369536, -0.5845446016898908, -0.4163843875261757, -0.25317662849243994, -0.1181181042361209, -0.029764045418275042, -1.9778607967202433e-05, -0.032858047193764284, -0.12389189692956848, -0.260846347871445, -0.424881121148646, -0.5926304694321702, -0.7387532281615223, -0.8387338060124853, -0.8716481906700546, -0.8226108448931253, -0.684648489624792, -0.45980230290679647, -0.1593355830685057, 0.19698796937922947, 0.5824981591492562, 0.9659402902161687, 1.3143079328245109, 1.5959168829854269, 1.7834246147509467, 1.856502203144232, 1.8038970021336898, 1.6246813255495194, 1.3285595114000848, 0.935195582414428, 0.4726175264772595, -0.025157111282938408, -0.5210547369520777, -0.978164898465893, -1.3629442189974066, -1.6481056037167963, -1.8149235684445624, -1.8547427504077514, -1.7695532909447826, -1.5715863657184777, -1.2819769618449777, -0.928629941935962, -0.5435008593073561, -0.1595575953081021, 0.19228257781144376, 0.48594614691525273, 0.7024612575967533, 0.8314695339517557, 0.871855924087331, 0.831448500989991, 0.7258322007291254, 0.5764077024646993, 0.40790048123150313, 0.24557826146976458, 0.11246247506834306, 0.02681824069671912, 0.0001779974963601605, 0.03609920110953435, 0.12978187566486266, 0.26858472287106794, 0.4333875587944874, 0.6006620981592641, 0.7450410429860499, 0.8421242169584686, 0.871235225455671, 0.8178453627997309, 0.6754129721778577, 0.44644314284993003, 0.14264896768803847, -0.2158159008805246, -0.6019916625348061, -0.9844655928223676, -1.3302236020321536, -1.607728588702458, -1.7899293277796937, -1.856907770295504, -1.79790531389617, -1.6125267248704085, -1.3109983508966736, -0.9134475090697709, -0.44826370361468204, 0.050309495703589, 0.5451290819063698, 0.9993782932619928, 1.379761784955804, 1.6593716647746888, 1.8199574729361576, 1.8533905678827423, 1.7621908481886355, 1.5590737523422789, 1.2655692740325644, 0.9098531925690092, 0.524005260458768, 0.1409624984378125, -0.20853711668183916, -0.4987269872798277, -0.7110369461897588, -0.8355635137875306, -0.8716536198471007, -0.827558297145423, -0.7192048792935392, -0.5682229835848305, -0.39943251209382746, -0.23805391397713127, -0.10692694458047258, -0.024021627459081762, -0.0004943820743091853, -0.039486412051518927, -0.13578602116645633, -0.2763890279315778, -0.44190056454724547, -0.6086362855354197, -0.7512098351782664, -0.8453434034808158, -0.8706143263194887, -0.8128553690799766, -0.6659591714626243, -0.43289514347611735, -0.1258242409392749, 0.23471404901851767, 0.6214762505178326, 1.0028987967470282, 1.3459676963015934, 1.619300533977318, 1.7961440584949404, 1.8569962049657192, 1.7915954505005136, 1.6000796663594317, 1.2931953955990347, 0.8915291213964807, 0.42382579710416146, -0.07545242743541863, -0.5691012598714213, -1.0204056796012408, -1.396325595103365, -1.6703380358406887, -1.8246712551908106, -1.8517246584631644, -1.754546758065331, -1.5463337733227, -1.2490052373522718, -0.8910009351661438, -0.5045171395249992, -0.12245230795708217, 0.2246418644385655, 0.5113113018704767, 0.7193912277836279, 0.8394345272832185, 0.8712492358099146, 0.8235062479244073, 0.712470315873728, 0.5599936602736069, 0.39098357560819025, 0.23060622289667465, 0.10151340507096407, 0.021375149382137915, 0.0009688259585469705, 0.04301853500213215, 0.14190227291641366, 0.2842565094153522, 0.45041699074564207, 0.6165498353213879, 0.7572567165311027, 0.8483891166719082, 0.8697841553831671, 0.8076406206097445, 0.6562880149455977, 0.4191603615067401, 0.10886443258310596, -0.25367866838146413, -0.640947796764441, -1.0212357751303314, -1.3615364778643189, -1.6306297312946463, -1.8020668671108684, -1.8567668195544753, -1.7849680674097383, -1.5873421160977976, -1.2751537684900516, -0.8694444379794453, -0.39930837628083693, 0.10058118272320174, 0.5929668034478223, 1.0412432332370969, 1.4126327950505306, 1.6810030686671764, 1.8290645973439354, 1.849746036007226, 1.7466232455220014, 1.5333696346725096, 1.2322887224495644, 0.8720773332695444, 0.48504056208577817, 0.10403061921539514, -0.24059401406651415, -0.5236973059726127, -0.7275234683124798, -0.8430831028210004, -0.8706443586878362, -0.8192947904015375, -0.7056315083458797, -0.5517229502484519, -0.3825567520809704, -0.223237794097897, -0.09622370546802472, -0.018879699133695693, -0.0016011696101817519, -0.04669437535071763, -0.1481285297150936, -0.29218438636103283, -0.458933678854628, -0.6243995582736543, -0.7631788232980969, -0.8512591473503794, -0.8687434259751498, -0.8022009319485268, -0.6464004885215058, -0.4052409069400953, -0.09177261503062162, 0.27270598604308716, 0.6604021656506354, 1.0394724113985545, 1.376926238280632, 1.6417132391273337, 1.8076958724725634, 1.8562189924610109, 1.7780238874246086, 1.574316102630959, 1.2568766443460582, 0.8471975137001779, 0.3747160278891797, -0.12569104091796438, -0.616721268567414, -1.061887171297632, -1.4286805859632756, -1.6913651795716926, -1.8331372491228102, -1.8474557787716193, -1.738422590849204, -1.5201845837483274, -1.2154236237546487, -0.8530865548039466, -0.4655795787379869, -0.08570099511055462, 0.2563908049159962, 0.5358832702990945, 0.7354330925382442, 0.8465098251028264, 0.8698406234340679, 0.8149263971290545, 0.6986914739591044, 0.5434140728201429, 0.3741551057801485, 0.2159512017253995, 0.09105965082708856, 0.01653611813069522, 0.0023912003779021918, 0.05051268921751395, 0.15446265025309103, 0.30016985124544804, 0.46744746033997253, 0.6321822730426403, 0.7689733170231692, 0.8539513267368916, 0.8674909030815849, 0.7965361755165627, 0.6362976363945707, 0.3911389426441606, 0.07455190268329948, -0.2917922024138119, -0.6798352132243723, -1.0576046002446677, -1.3921332993439268, -1.6525481626762528, -1.8130292525581742, -1.8553521682982697, -1.7707637005864436, -1.5610037165668218, -1.2383672490663655, -0.8247924388593862, -0.35005335508028335, 0.15077728551140837, 0.6403602354624879, 1.0823337530990695, 1.4444662251454923, 1.7014228497348225, 1.8368890278340615, 1.8448550290925256, 1.729947129088558, 1.5067819084439997, 1.1984138585408113, 0.8340327710915711, 0.446138224162265, 0.06746696529607055, -0.2720295232799653, -0.5478675212991575, -0.7431195840651319, -0.849715334873547, -0.8688397127065085, -0.8104035753981083, -0.6916532484713249, -0.5350702479922234, -0.3657816840903337, -0.20874898749543924, -0.08602300184024758, -0.014345196311190911, -0.0033386525555608726, -0.05447218379098173, -0.16090245369417444, -0.3082100707531341, -0.47595515754620965, -0.6398948070696269, -0.7746373853647275, -0.8564635271201817, -0.8660254037844438, -0.7906462817573696, -0.6259805609450265, -0.3768566839356007, -0.05720545126182952, 0.31093349209975457, 0.6992427881709042, 1.075628248607509, 1.4071540139788905, 1.6631316545997776, 1.818065244967709, 1.8541658580916747, 1.763188364064255, 1.5474071101594253, 1.2196288589907591, 0.802233338291176, 0.3253249764052183, -0.1758352051691251, -0.6638793096291262, -1.102579280949191, -1.4599870266079589, -1.7111746254827898, -1.8403198183354834, -1.841944993051891, -1.721199249428043, -1.4931649363751904, -1.1812633659783667, -0.8149201558674426, -0.42672051619464785, -0.04933202539771919, 0.2875075029588475, 0.5596484414529617, 0.7505824853392168, 0.8527003286300973, 0.8676433563197363, 0.805728866491914, 0.6845198852813366, 0.5266946955613105, 0.3574395166731309, 0.2016336600017632, 0.08111547435794086, 0.012307671920328, 0.004443207454131626, 0.058571517678495555, 0.167445720269215, 0.3163021865532996, 0.4844535845777672, 0.6475339974822187, 0.7801682429137391, 0.8587936625132271, 0.8643457976868273, 0.7845312392858612, 0.6154504225807558, 0.362396398145315, 0.03973645712413765, -0.33012600476877235, -0.7186207327808236, -1.0935392766488297, -1.4219847671329882, -1.6734609157326195, -1.8228021473982423, -1.8526596394626333, -1.755298802026348, -1.5335284968782457, -1.2006648002052114, -0.7795243704692361, -0.300535524804923, 0.20086009476270286, 0.6872741227845326, 1.1226201009406847, 1.4752403616240835, 1.7206191185549724, 1.843429572992188, 1.8387269401291635, 1.7121813945852506, 1.4793370340555496, 1.1639761061845983, 0.7957528842954324, 0.4073304549034416, 0.03129963623941823, -0.3028221258128799, -0.5712244695517152, -0.7578213976331398, -0.8554655583167261, -0.8662533306853192, -0.8009048449307925, -0.6772944545570402, -0.5182906342180273, -0.3491316146326188, -0.19460769403051803, -0.07633873892282483, -0.010424231310329546, -0.005704493488013718, -0.06280930127020146, -0.17409019188080077, -0.3244433160839539, -0.4929395481829225, -0.655096691988154, -0.7855631320057122, -0.8609396892993911, -0.8624510073253329, -0.7781910950216723, -0.6047084395741031, -0.34776040417023935, -0.022148156572202704, 0.34936586602408004, 0.7379648839211274, 1.1113336187286658, 1.4366219766615946, 1.683533195794222, 1.8272383181054535, 1.8508331567966474, 1.7470960054961848, 1.5193701509631103, 1.1814784478358318, 0.7566697266048852, 0.27568964659640677, -0.22584725640043438, -0.7105403338183975, -1.142452603734298, -1.49022365927233, -1.7297550063560325, -1.8462183116171302, -1.8352022028384836, -1.7028960601785004, -1.4653016060643502, -1.1465560592695425, -0.7765351319849121, -0.3879720216715723, -0.01337322307866784, 0.31797082230181706, 0.5825941009634175, 0.7648359810159188, 0.8580118310065448, 0.8646714582409591, 0.7959341177093664, 0.6699800423600548, 0.509861280649058, 0.34086096968652796, 0.18767352988475228, 0.07169442031603956, 0.008695508754549941, 0.007122086275669348, 0.06718409711610274, 0.18083357271854716, 0.3326305533430836, 0.5014098486405265, 0.6625797497670529, 0.7908193235261399, 0.8628996068682364, 0.8603400085692348, 0.771625954308075, 0.5937558878840322, 0.3329510720119815, 0.004443825148480074, -0.36864917828446186, -0.7572710740084568, -1.1290072243780496, -1.4510620942059593, -1.6933457940861274, -1.8313721763514064, -1.8486861213960166, -1.7385810321929074, -1.5049344069649506, -1.1620732253313981, -0.7336736297377222, -0.2507920004561957, 0.2507920004561623, 0.7336736297376916, 1.1620732253313728, 1.5049344069649329, 1.7385810321928985, 1.848686121396017, 1.831372176351416, 1.6933457940861452, 1.451062094205983, 1.1290072243780491, 0.7572710740084577, 0.3686491782844642, -0.004443825148476188, -0.33295107201197616, -0.5937558878840259, -0.7716259543080676, -0.8603400085692265, -0.8628996068682278, -0.7908193235261313, -0.6625797497670447, -0.5014098486405191, -0.33263055334307745, -0.1808335727185425, -0.06718409711609985, -0.007122086275668571, -0.00869550875455094, -0.07169442031604256, -0.18767352988475705, -0.34086096968650886, -0.5098612806490372, -0.669980042360035, -0.7959341177093499, -0.8646714582409478, -0.8580118310065404, -0.764835981015922, -0.5825941009634286, -0.3179708223018355, 0.013373223078643748, 0.38797202167154443, 0.7765351319848831, 1.146556059269515, 1.4653016060643267, 1.702896060178483, 1.8352022028384747, 1.8462183116171302, 1.729755006356042, 1.490223659272348, 1.142452603734303, 0.7105403338184029, 0.22584725640043976, -0.2756896465964014, -0.7566697266048799, -1.1814784478358265, -1.5193701509631055, -1.7470960054961804, -1.8508331567966434, -1.8272383181054503, -1.6835331957942192, -1.436621976661593, -1.1113336187286655, -0.7379648839211284, -0.34936586602408254, 0.022148156572198707, 0.3477604041702341, 0.6047084395740965, 0.7781910950216647, 0.862451007325338, 0.860939689299403, 0.7855631320057291, 0.655096691988174, 0.49293954818294305, 0.32444331608397275, 0.17409019188081587, 0.062809301270211, 0.005704493488016715, 0.01042423131032566, 0.0763387389228144, 0.19460769403050227, 0.3491316146325996, 0.5182906342180068, 0.6772944545570205, 0.8009048449307763, 0.8662533306853082, 0.8554655583167221, 0.7578213976331435, 0.5712244695517217, 0.3028221258128849, -0.03129963623941445, -0.4073304549034394, -0.7957528842954315, -1.1639761061845988, -1.4793370340555512, -1.7121813945852533, -1.8387269401291668, -1.8434295729921903, -1.7206191185549795, -1.475240361624095, -1.1226201009407, -0.6872741227845507, -0.20086009476272232, 0.3005355248048897, 0.7795243704691923, 1.2006648002051867, 1.5335284968782177, 1.7552988020263307, 1.852659639462635, 1.822802147398248, 1.673460915732636, 1.4219847671330124, 1.0935392766488612, 0.7186207327808527, 0.33012600476880805, -0.03973645712410412, -0.3623963981452918, -0.6154504225807309, -0.784531239285848, -0.8643457976868397, -0.8587936625132357, -0.7801682429137693, -0.6475339974822529, -0.4844535845777913, -0.31630218655333076, -0.1674457202692139, -0.05857151767849522, -0.004443207454132514, -0.012307671920328112, -0.08111547435794142, -0.20163366000175775, -0.3574395166731334, -0.5266946955613041, -0.6845198852813433, -0.8057288664919238, -0.8676433563197257, -0.8527003286301077, -0.7505824853392347, -0.5596484414529737, -0.28750750295887806, 0.049332025397694546, 0.42672051619461976, 0.8149201558674083, 1.1812633659783391, 1.4931649363751922, 1.7211992494280457, 1.8419449930518892, 1.8403198183354874, 1.7111746254827942, 1.4599870266079522, 1.102579280949196, 0.6638793096291178, 0.17583520516911627, -0.3253249764052129, -0.8022333382911719, -1.219628858990757, -1.5474071101594147, -1.763188364064258, -1.8541658580916738, -1.8180652449677175, -1.6631316545997876, -1.407154013978902, -1.0756282486075195, -0.6992427881708856, -0.31093349209974064, 0.05720545126184018, 0.3768566839356011, 0.6259805609450281, 0.790646281757373, 0.8660254037844284, 0.8564635271201695, 0.774637385364719, 0.6398948070696105, 0.47595515754620754, 0.30821007075313644, 0.160902453694173, 0.05447218379098262, 0.0033386525555602065, 0.014345196311191466, 0.08602300184024625, 0.20874898749544424, 0.3657816840903357, 0.5350702479922114, 0.691653248471307, 0.8104035753981182, 0.8688397127065146, 0.8497153348735494, 0.7431195840651379, 0.5478675212991604, 0.2720295232799661, -0.06746696529605156, -0.44613822416224613, -0.8340327710915422, -1.1984138585407853, -1.5067819084439769, -1.7299471290885446, -1.8448550290925203, -1.8368890278340628, -1.7014228497348283, -1.4444662251455058, -1.0823337530991088, -0.6403602354625001, -0.15077728551141376, 0.350053355080278, 0.8247924388593877, 1.2383672490663726, 1.561003716566806, 1.7707637005864347, 1.8553521682982659, 1.8130292525581733, 1.6525481626762621, 1.3921332993439395, 1.0576046002446835, 0.6798352132243874, 0.29179220241382664, -0.07455190268329126, -0.3911389426441617, -0.636297636394556, -0.7965361755165512, -0.8674909030815838, -0.8539513267369006, -0.7689733170231802, -0.6321822730426344, -0.46744746033997403, -0.30016985124545414, -0.15446265025309602, -0.050512689217518614, -0.002391200377903635, -0.016536118130694333, -0.09105965082707945, -0.21595120172538307, -0.37415510578012423, -0.5434140728201112, -0.6986914739590977, -0.8149263971290457, -0.8698406234340578, -0.8465098251028162, -0.7354330925382422, -0.5358832702991201, -0.25639080491600674, 0.0857009951105463, 0.46557957873798483, 0.8530865548039485, 1.2154236237546518, 1.5201845837483294, 1.7384225908492081, 1.8474557787716228, 1.8331372491228106, 1.6913651795716955, 1.4286805859632996, 1.0618871712976545, 0.6167212685674325, 0.12569104091798394, -0.3747160278891676, -0.8471975137001739, -1.2568766443460377, -1.5743161026309433, -1.7780238874246033, -1.8562189924610135, -1.8076958724725696, -1.6417132391273408, -1.3769262382806444, -1.0394724113985687, -0.6604021656506505, -0.2727059860430977, 0.09177261503061884, 0.4052409069400784, 0.6464004885214905, 0.8022009319485259, 0.8687434259751634, 0.8512591473504065, 0.7631788232981016, 0.6243995582736687, 0.4589336788546485, 0.29218438636105504, 0.14812852971511048, 0.04669437535071852, 0.0016011696101809747, 0.018879699133698247, 0.09622370546802816, 0.22323779409789857, 0.3825567520809678, 0.5517229502484706, 0.7056315083458944, 0.8192947904015462, 0.8706443586878378, 0.8430831028209961, 0.7275234683124998, 0.5236973059726286, 0.24059401406652658, -0.1040306192153807, -0.48504056208576674, -0.8720773332695345, -1.2322887224495491, -1.5333696346724994, -1.7466232455219952, -1.8497460360072215, -1.829064597343935, -1.6810030686671946, -1.4126327950505493, -1.0412432332371186, -0.5929668034478475, -0.10058118272322131, 0.3993083762808386, 0.8694444379794534, 1.2751537684900647, 1.587342116097809, 1.784968067409734, 1.8567668195544749, 1.8020668671108733, 1.630629731294651, 1.3615364778643224, 1.0212357751303385, 0.6409477967644461, 0.2536786683814687, -0.10886443258312584, -0.4191603615067605, -0.656288014945619, -0.8076406206097385, -0.8697841553831581, -0.8483891166719243, -0.7572567165311004, -0.6165498353214117, -0.4504169907456212, -0.2842565094153622, -0.14190227291640334, -0.04301853500213815, -0.0009688259585465264, -0.0213751493821297, -0.10151340507097151, -0.23060622289665966, -0.39098357560819413, -0.5599936602735766, -0.7124703158737212, -0.8235062479244458, -0.8712492358099205, -0.8394345272832286, -0.7193912277836183, -0.5113113018704564, -0.22464186443857037, 0.12245230795710327, 0.504517139525001, 0.8910009351661682, 1.2490052373522678, 1.546333773322719, 1.75454675806533, 1.8517246584631741, 1.8246712551908102, 1.6703380358406799, 1.396325595103363, 1.0204056796012382, 0.5691012598714401, 0.0754524274354027, -0.4238257971041529, -0.8915291213965006, -1.2931953955990392, -1.6000796663594268, -1.7915954505005103, -1.8569962049657165, -1.796144058494953, -1.619300533977312, -1.3459676963016125, -1.0028987967470213, -0.6214762505178515, -0.234714049018508, 0.12582424093927058, 0.4328951434761301, 0.6659591714626163, 0.8128553690799722, 0.8706143263195066, 0.8453434034808209, 0.7512098351782475, 0.6086362855354087, 0.441900564547211, 0.27638902793157594, 0.13578602116644323, 0.03948641205152292, 0.0004943820743085192, 0.02402162745908154, 0.1069269445804868, 0.2380539139771256, 0.3994325120938448, 0.5682229835848142, 0.7192048792935714, 0.8275582971454223, 0.8716536198471109, 0.8355635137875196, 0.7110369461897494, 0.4987269872798407, 0.20853711668183583, -0.1409624984378003, -0.5240052604587849, -0.9098531925689993, -1.265569274032582, -1.5590737523422633, -1.762190848188642, -1.8533905678827358, -1.8199574729361538, -1.6593716647746866, -1.379761784955808, -0.9993782932620148, -0.5451290819063649, -0.05030949570360859, 0.44826370361469065, 0.9134475090697588, 1.3109983508966654, 1.612526724870393, 1.79790531389617, 1.8569077702955137, 1.7899293277796975, 1.6077285887024355, 1.3302236020321592, 0.9844655928223454, 0.6019916625348095, 0.21581590088050107, -0.14264896768804392, -0.44644314284992725, -0.6754129721778539, -0.8178453627997251, -0.8712352254556811, -0.8421242169584658, -0.7450410429860499, -0.600662098159244, -0.4333875587944999, -0.2685847228710584, -0.1297818756648771, -0.036099201109528134, -0.00017799749636038253, -0.026818240696723117, -0.11246247506833085, -0.24557826146976725, -0.4079004812314756, -0.5764077024647165, -0.7258322007291116, -0.8314485009899963, -0.8718559240873135, -0.8314695339517495, -0.7024612575967683, -0.4859461469152618, -0.1922825778114653, 0.15955759530810076, 0.5435008593073304, 0.9286299419359638, 1.281976961844947, 1.5715863657184737, 1.7695532909447629, 1.8547427504077492, 1.8149235684445588, 1.6481056037168107, 1.3629442189974044, 0.9781648984659053, 0.5210547369520592, 0.025157111282943793, -0.4726175264772835, -0.935195582414403, -1.3285595114000857, -1.624681325549512, -1.8038970021336909, -1.8565022031442395, -1.783424614750936, -1.5959168829854402, -1.3143079328245029, -0.9659402902161901, -0.5824981591492473, -0.1969879693792601, 0.15933558306849782, 0.4598023029067806, 0.6846484896247883, 0.8226108448931347, 0.8716481906700604, 0.8387338060124907, 0.7387532281615122, 0.5926304694321939, 0.4248811211486473, 0.26084634787147165, 0.12389189692956337, 0.03285804719377183, 1.9778607967202433e-05, 0.029764045418265273, 0.1181181042361168, 0.25317662849240785, 0.41638438752617934, 0.5845446016898619, 0.7323493025369491, 0.8351744585914136, 0.8718546105842784, 0.8271521159763798, 0.6936648548689436, 0.4729706205956447, 0.17588110061849305, -0.17823397156837006, -0.5629998564463109, -0.9473270250923809, -1.2982244548656414, -1.5838684495799933, -1.7766319173099445, -1.8557802568827086, -1.8095699271693526, -1.636541565166966, -1.3458758065781478, -0.9567693600110552, -0.4968827147647287, -1.959434878635765e-14};
  size_t n_points = sizeof(sample)/sizeof(cufftReal);

  // create plan for performing fft
  cufftHandle plan;
  if (cufftPlan1d(&plan, n_points, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
    printf("Failed to create 1D plan\n");
    return nullptr;
  }

  // load data to gpu
  cufftReal *idata;
  cudaMalloc((void**) &idata, sizeof(cufftComplex)*n_points);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory space for input data.\n");
    return nullptr;
  }

  cudaMemcpy(idata, sample, sizeof(sample)/sizeof(double), cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to load time data to memory.\n");
    return nullptr;
  }

  // prepare memory for return data
  cufftComplex *odata;
  cudaMalloc((void**) &odata, sizeof(cufftComplex)*(n_points/2 + 1));
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory for output data.\n");
  }

  // perform fft
  if (cufftExecR2C(plan, idata, odata) != CUFFT_SUCCESS) {
    printf("Failed to perform fft.\n");
    return nullptr;
  }

  // grab data from graphics and print (memcpy waits until complete) cuda memcopy doesn't complete
  // can return errors from previous cuda calls if they haven't been caught
  cufftComplex *out_sample;
  size_t num_bytes = (n_points/2 + 1)*sizeof(cufftComplex);
  out_sample = new cufftComplex[n_points/2 + 1];
  cudaMemcpy(out_sample, odata, num_bytes, cudaMemcpyDeviceToHost);
  int error_value = cudaGetLastError();
  printf("cudaMemcpy from device state: %i\n", error_value);
  if(error_value != cudaSuccess) {
    printf("Failed to pull data from device.\n");
    return nullptr;
  }

  // adjust magnitude of fourier coefficiencts appropriately
  float * fft_magnitude = new float[n_points/2+1];
  for (size_t i = 0; i < n_points/2 + 1; i++) {
      fft_magnitude[i] = 2*sqrt(out_sample[i].x*out_sample[i].x +
                         out_sample[i].y*out_sample[i].y) / n_points;
  }

  fft_magnitude[0] /= 2;

  // print out for matlab review
  printf("fft_magnitude = [");
  for (size_t i = 0; i < n_points/2 + 1; i++) {
    printf("%f, ", fft_magnitude[i]);
  }
  printf("\b\b]\n");

  // clean up
  delete(out_sample);
  cufftDestroy(plan);
  cudaFree(idata);

  return fft_magnitude;
}
*/

/******************************************************************************/


bool load_video(std::string video_source, cufftReal * d_mat, video_param_t * video_param) {

  // grab video
  cv::VideoCapture captRefrnc(video_source);
  if (!captRefrnc.isOpened()) {

    printf("Could not grab %s\n", video_source.c_str());
    return -1;
  }

  // allocate gpu memory
  cudaMalloc((void**) &d_mat, sizeof(cv::Mat) * video_param->n_frames);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory space for video data.\n");
    return -1;
  }

  // load to gpu
  size_t d_mat_size = sizeof(cufftReal) * video_param->n_frames;
  cudaMemcpy(d_mat, (const void *)captRefrnc, d_mat_size, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to load video data to memory.\n");
    return -1;
  }

  return 0;
}


/******************************************************************************/


int main(int argc, char** argv) {

  // setup processing parameters
  int heliostat_freq[] = {1, 2, 3};
  if (argc == 0) {
    printf("Using default video.\n");
  }


  std::string video_source = "vib.avi";
  video_param_t video_param;

  cv::Mat * d_mat;

  cufftReal * d_raw;
  size_t n_threads = 2432;
  int n_pixel_vals = video_param.height *
                    video_param.width *
                    video_param.n_frames;
  size_t n_blocks =  n_pixel_vals / n_threads;
  cufftComplex * d_ftd;
  cufftReal * d_mag;

  cv::Mat images[sizeof(heliostat_freq)/sizeof(int)];

  while(true) {

    // @will need to update depending on rolling buffer or large batch type
    // rolling will not be in-place

    if (load_video(video_source, d_mat, &video_param) != 0) {

      printf("Could not load video data.\n");
      return -1;
    }

    // prepare for fft @depend on config of gpu
    // cufftReal_convert<<<n_blocks, n_threads>>>(d_mat, d_raw);

    // perform fft on individual pixel streams and adjust to real values
    cudaMalloc((void**) &d_raw, sizeof(cufftComplex)*n_points * batch);
    if (cudaGetLastError() != cudaSuccess) {
      printf("Failed to allocate memory space for video data.\n");
      return -1;
    }

    if (fft_batched(d_raw, video_param, d_ftd)) {

      printf("Could not perform fft.\n");
      return -1;
    }
    // mag_adjust(d_ftd, d_mag);

    // grab bin values for specific frequencies across entire stream
    // @put into cuda func
    // image_collate(heliostat_freq, images);

    // perform centroid calculations (in place)
    heliostat_dev_t heliostat_dev[3];
    // centroid_calc(images, heliostat_dev);

    send_dev_rpi(heliostat_dev);

  }

  // float * fft_magnitude = fft1d();

  // clean up
  // delete(fft_magnitude);

  return 0;
}
