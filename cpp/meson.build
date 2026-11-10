project('hcpose_postprocess', 'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++17', 'buildtype=release', 'warning_level=1',
                     'cpp_args=-Wno-psabi -Wno-deprecated-declarations'])

# TAPPAS core headers (hailo_objects.hpp, hailo_common.hpp). The package was
# renamed across releases, so try every known name.
tappas_dep = dependency('hailo-tappas-core', required : false)
if not tappas_dep.found()
  tappas_dep = dependency('hailo-apps-core', required : false)
endif
if not tappas_dep.found()
  tappas_dep = dependency('hailo_tappas', required : false)
endif

extra_inc = []
extra_args = []
if not tappas_dep.found()
  # Fallback for local builds/tests: -Dtappas_include=/path/to/core/hailo/general
  tappas_inc = get_option('tappas_include')
  if tappas_inc == ''
    error('hailo-tappas-core not found via pkg-config. On Raspberry Pi OS run: sudo apt install hailo-all')
  endif
  extra_inc += include_directories(tappas_inc, is_system : true)
endif

hailort_inc = get_option('hailort_include')
if hailort_inc != ''
  extra_inc += include_directories(hailort_inc, is_system : true)
endif

# rapidjson: header-only; TAPPAS exposes its copy through pkg-config,
# otherwise use the system one (apt install rapidjson-dev).
rapidjson_dir = ''
if tappas_dep.found()
  rapidjson_dir = tappas_dep.get_variable(pkgconfig : 'rapidjson_includedir', default_value : '')
endif
if rapidjson_dir == ''
  rapidjson_dir = get_option('rapidjson_include')
endif
if rapidjson_dir != ''
  extra_inc += include_directories(rapidjson_dir, is_system : true)
else
  rj = dependency('RapidJSON', required : false)
  if not rj.found()
    rj = dependency('rapidjson', required : false)
  endif
  if not rj.found()
    error('rapidjson headers not found. Install with: sudo apt install rapidjson-dev')
  endif
  tappas_dep = [tappas_dep, rj]
endif

shared_library('hcpose_yolov8_pose',
  ['yolov8_pose_postprocess.cpp', 'tiling.cpp'],
  dependencies : tappas_dep,
  include_directories : extra_inc,
  gnu_symbol_visibility : 'default',
  install : false,
)
