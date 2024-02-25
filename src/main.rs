use std::{f32::consts::PI, rc::Rc};

use glam::{vec3, Vec3};
use rand::random;

const RATION: f32 = 16. / 9.;
const WIDTH: usize = 1200;
const HEIGHT: usize = (WIDTH as f32 / RATION) as usize;
// const VIEWPORT_HEIGHT: f32 = 2.;
// const VIEWPORT_WIDTH: f32 = VIEWPORT_HEIGHT * (WIDTH as f32 / HEIGHT as f32);

type Color = Vec3;
fn write_pixel(buffer: &mut [u8], x: usize, y: usize, color: Color, samples_per_pixel: usize) {
    let scale = 1. / samples_per_pixel as f32;
    buffer[(y * WIDTH + x) * 3] = ((color.x * scale).sqrt().clamp(0., 0.999) * 256.) as u8;
    buffer[(y * WIDTH + x) * 3 + 1] = ((color.y * scale).sqrt().clamp(0., 0.99) * 256.) as u8;
    buffer[(y * WIDTH + x) * 3 + 2] = ((color.z * scale).sqrt().clamp(0., 0.999) * 256.) as u8;
}

fn ray_color(ray: Ray, hittable: &[Box<dyn Hittable>], depth: i32) -> Color {
    if depth < 0 {
        return Color::new(1., 0., 0.);
    }
    let mut hit_record = Record::default();
    let mut hit_anything = false;
    for e in hittable {
        if e.hit(ray, 0.001, hit_record.t, &mut hit_record) {
            hit_anything = true;
        }
    }
    if hit_anything {
        let mut ray_scatter = Ray::default();
        let mut color = Color::ZERO;
        return if hit_record
            .material
            .scatter(ray, &hit_record, &mut color, &mut ray_scatter)
        {
            color * ray_color(ray_scatter, hittable, depth - 1)
        } else {
            Color::ZERO
        };
    }
    let t = 0.5 * (ray.dir.y + 1.);
    (1. - t) * Color::new(1., 1., 1.) + t * vec3(0.5, 0.7, 1.)
}

#[derive(Clone, Copy, Default)]
struct Ray {
    point: Vec3,
    dir: Vec3,
}
impl Ray {
    fn at(&self, t: f32) -> Vec3 {
        self.point + t * self.dir
    }
}
struct Sphere {
    center: Vec3,
    radius: f32,
    material: Material,
}

struct Record {
    point: Vec3,
    normal: Vec3,
    t: f32,
    front_face: bool,
    material: Material,
}
#[derive(Clone)]
enum Material {
    Empty,
    Lambertian(Color),
    Metal(Color, f32),
    Dielectric(f32),
}

impl Material {
    fn scatter(
        &self,
        ray_in: Ray,
        record: &Record,
        attenuation: &mut Color,
        ray_scatter: &mut Ray,
    ) -> bool {
        match self {
            Material::Lambertian(color) => {
                // 次表面散射
                let scatter_dir = record.normal + random_in_unit_sphere();
                ray_scatter.point = record.point;
                ray_scatter.dir = scatter_dir.normalize();
                *attenuation = *color;
                true
            }
            Material::Metal(color, fuzz) => {
                // 镜面反射
                let reflected = ray_in.dir - 2. * ray_in.dir.dot(record.normal) * record.normal;
                ray_scatter.point = record.point;
                ray_scatter.dir = (reflected + *fuzz * random_in_unit_sphere()).normalize();
                *attenuation = *color;
                true
            }
            Material::Dielectric(ir) => {
                *attenuation = Vec3::ONE;
                let refraction_ration = if record.front_face { 1. / *ir } else { *ir };
                let cos_theta = (-ray_in.dir.dot(record.normal)).min(1.);
                let sin_theta = (1. - cos_theta * cos_theta).sqrt();
                let cannot_refract = refraction_ration * sin_theta > 1.;
                let refracted =
                    if cannot_refract || reflectance(cos_theta, refraction_ration) > random() {
                        ray_in.dir - 2. * ray_in.dir.dot(record.normal) * record.normal
                    } else {
                        refract(ray_in.dir, record.normal, refraction_ration)
                    };
                ray_scatter.point = record.point;
                ray_scatter.dir = refracted.normalize();
                true
            }
            Material::Empty => false,
        }
    }
}
impl Default for Record {
    fn default() -> Self {
        let record = Record {
            point: Vec3::NAN,
            normal: Vec3::NAN,
            t: f32::INFINITY,
            front_face: false,
            material: Material::Empty,
        };
        record
    }
}
impl Record {
    fn set_face_normal(&mut self, ray: Ray, outward_normal: Vec3) {
        self.front_face = ray.dir.dot(outward_normal) < 0.;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}
trait Hittable {
    fn hit(&self, ray: Ray, min: f32, max: f32, record: &mut Record) -> bool;
}

impl Hittable for Sphere {
    fn hit(&self, ray: Ray, min: f32, max: f32, record: &mut Record) -> bool {
        let oc = ray.point - self.center;
        let a = ray.dir.length_squared();
        let half_b = ray.dir.dot(oc);
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0. {
            return false;
        }
        let sqrt = discriminant.sqrt();
        let solution_near = (-half_b - sqrt) / a;
        let solution_far = (-half_b + sqrt) / a;
        let mut solution = solution_near;
        if solution < min || solution > max {
            solution = solution_far;
            if solution < min || solution > max {
                return false;
            }
        }
        record.t = solution;
        record.point = ray.at(solution);
        let outward_normal = (record.point - self.center).normalize();
        record.set_face_normal(ray, outward_normal);
        record.material = self.material.clone();
        true
    }
}

fn random_in_unit_sphere() -> Vec3 {
    loop {
        let res = vec3(
            random::<f32>() * 2. - 1.,
            random::<f32>() * 2. - 1.,
            random::<f32>() * 2. - 1.,
        );
        if res.length_squared() < 1. {
            return res.normalize();
        }
    }
}
fn random_in_unit_disk() -> Vec3 {
    loop {
        let res = vec3(random::<f32>() * 2. - 1., random::<f32>() * 2. - 1., 0.);
        if res.length_squared() < 1. {
            return res.normalize();
        }
    }
}

fn refract(uv: Vec3, n: Vec3, ration: f32) -> Vec3 {
    let cos_theta = ((-uv).dot(n)).min(1.);
    let r_out_perp = ration * (uv + cos_theta * n);
    let r_out_parallel = -((1. - r_out_perp.length_squared()).abs()).sqrt() * n;
    r_out_perp + r_out_parallel
}
fn reflectance(cos: f32, idx: f32) -> f32 {
    let mut r0 = (1. - idx) / (1. + idx);
    r0 = r0 * r0;
    r0 + (1. - r0) * ((1. - cos).powf(5.))
}
fn random_color() -> Vec3 {
    Vec3::new(random(), random(), random())
}
fn main() {
    let mut frame_buffer: Vec<u8> = vec![255; WIDTH * HEIGHT * 3];

    // camera
    let samples_per_pixel = 500;
    let vfov = 20;
    let look_from = Vec3::new(13., 2., 3.);
    let look_at = Vec3::new(0., 0., 0.);
    let up = Vec3::new(0., 1., 0.);
    let (u, v, w): (Vec3, Vec3, Vec3);
    let defocus_angel = 0.6;
    let focus_dis = 10.;
    let defocus_disk_u;
    let defocus_disk_v;

    let camera_center = look_from;
    w = (look_from - look_at).normalize();
    u = up.cross(w).normalize();
    v = w.cross(u);

    let theta = vfov as f32 * PI / 180.;
    let h = (theta / 2.).tan();
    let viewport_height = 2. * h * focus_dis;
    let viewport_width = viewport_height * (WIDTH as f32 / HEIGHT as f32);

    let viewport_u = viewport_width * u;
    let viewport_v = viewport_height * -v;

    let pixel_delta_u = viewport_u / WIDTH as f32;
    let pixel_delta_v = viewport_v / HEIGHT as f32;

    let pixel_sample_square = || -> Vec3 {
        let px = -0.5 + rand::random::<f32>();
        let py = -0.5 + rand::random::<f32>();
        px * pixel_delta_u + py * pixel_delta_v
    };

    let viewport_upper_left = camera_center - focus_dis * w - viewport_u / 2. - viewport_v / 2.;

    let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    let defocus_radius = focus_dis * (((defocus_angel / 2.) * PI / 180.).tan());
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;

    // world init
    let mut world: Vec<Box<dyn Hittable>> = vec![];
    // ground
    world.push(Box::new(Sphere {
        center: vec3(0., -1000.5, -1.),
        radius: 1000.,
        material: Material::Lambertian(vec3(0.5, 0.5, 0.5)),
    }));
    for i in 0..22 {
        for j in 0..22 {
            let r = random::<f32>();
            let center = Vec3::new(
                (i - 11) as f32 + 0.8 * random::<f32>(),
                0.2,
                (j - 11) as f32 + 0.9 * random::<f32>(),
            );
            if (center - Vec3::new(4., 0.2, 0.)).length() > 0.9 {
                if r < 0.8 {
                    let color = random_color();
                    let material = Material::Lambertian(color);
                    world.push(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        material,
                    }))
                } else if r < 0.95 {
                    let color = (random_color() + 0.5) / 1.5;
                    let fuzz = (random::<f32>() + 0.5) / 1.5;
                    let material = Material::Metal(color, fuzz);
                    world.push(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        material,
                    }))
                } else {
                    let material = Material::Dielectric(1.5);
                    world.push(Box::new(Sphere {
                        center,
                        radius: 0.2,
                        material,
                    }))
                }
            }
        }
    }
    world.push(Box::new(Sphere {
        center: vec3(0., 1., 0.),
        radius: 1.,
        material: Material::Dielectric(1.5),
        // material: Material::Dielectric(1.5),
    }));
    world.push(Box::new(Sphere {
        center: vec3(-4., 1., 0.),
        radius: 1.,
        material: Material::Lambertian(vec3(0.4, 0.2, 0.1)),
    }));
    world.push(Box::new(Sphere {
        center: vec3(4., 1., 0.),
        radius: 1.,
        material: Material::Metal(vec3(0.7, 0.6, 0.5), 0.0),
    }));

    for px in 0..WIDTH {
        for py in 0..HEIGHT {
            let mut color = Color::ZERO;
            for _ in 0..samples_per_pixel {
                let pixel_center =
                    pixel00_loc + (px as f32 * pixel_delta_u) + (py as f32 * pixel_delta_v);
                let pixle_sample = pixel_center + pixel_sample_square();
                let ray_origin = if defocus_angel <= 0. {
                    camera_center
                } else {
                    let p = random_in_unit_disk();
                    camera_center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v)
                };

                let ray_dir = (pixle_sample - ray_origin).normalize();
                let ray = Ray {
                    dir: ray_dir,
                    point: ray_origin,
                };
                color += ray_color(ray, &mut world, 50);
            }
            write_pixel(&mut frame_buffer, px, py, color, samples_per_pixel)
        }
    }
    image::save_buffer(
        "ray-tracy.png",
        &frame_buffer,
        WIDTH as u32,
        HEIGHT as u32,
        image::ColorType::Rgb8,
    )
    .unwrap();
}
