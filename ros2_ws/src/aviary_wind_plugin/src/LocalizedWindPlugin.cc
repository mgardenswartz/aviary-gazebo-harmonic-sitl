#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/ExternalWorldWrenchCmd.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/math/Vector3.hh>

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace aviary
{
    struct WindZone {
        std::string name;
        gz::math::Vector3d center;
        gz::math::Vector3d core_size;
        double buffer;
        gz::math::Vector3d force;
        double noise_std_dev;
    };

    class LocalizedWindPlugin
        : public gz::sim::System,
          public gz::sim::ISystemConfigure,
          public gz::sim::ISystemPreUpdate
    {
    public:
        LocalizedWindPlugin() : gen(1337) {}

        void Configure(const gz::sim::Entity &_entity,
                       const std::shared_ptr<const sdf::Element> &_sdf,
                       gz::sim::EntityComponentManager &_ecm,
                       gz::sim::EventManager &) override
        {
            if (_sdf->HasElement("target_model"))
                this->targetModelName = _sdf->Get<std::string>("target_model");
            if (_sdf->HasElement("target_link"))
                this->targetLinkName = _sdf->Get<std::string>("target_link");

            // Dynamically parse all <wind_zone> blocks from the SDF
            auto sdfClone = _sdf->Clone();
            if (sdfClone->HasElement("wind_zone")) {
                sdf::ElementPtr zoneElem = sdfClone->GetElement("wind_zone");
                while (zoneElem) {
                    WindZone zone;
                    zone.name = zoneElem->Get<std::string>("name", "Unnamed Fan").first;
                    zone.center = zoneElem->Get<gz::math::Vector3d>("center", gz::math::Vector3d::Zero).first;
                    zone.core_size = zoneElem->Get<gz::math::Vector3d>("core_size", gz::math::Vector3d(1.5, 1.5, 1.5)).first;
                    zone.buffer = zoneElem->Get<double>("buffer", 1.0).first;
                    zone.force = zoneElem->Get<gz::math::Vector3d>("force", gz::math::Vector3d::Zero).first;
                    zone.noise_std_dev = zoneElem->Get<double>("noise_std_dev", 0.0).first;
                    
                    this->zones.push_back(zone);
                    gzmsg << "[Aviary Wind] Loaded Zone: " << zone.name << " at " << zone.center << std::endl;
                    
                    zoneElem = zoneElem->GetNextElement("wind_zone");
                }
            } else {
                gzerr << "[Aviary Wind] No <wind_zone> elements found in SDF!" << std::endl;
            }
        }

        void PreUpdate(const gz::sim::UpdateInfo &_info,
                       gz::sim::EntityComponentManager &_ecm) override
        {
            if (_info.paused || _info.iterations == 0) return;

            // Bind to Link
            if (this->linkEntity == gz::sim::kNullEntity) {
                gz::sim::Entity modelEntity = _ecm.EntityByComponents(
                    gz::sim::components::Name(this->targetModelName),
                    gz::sim::components::Model()
                );
                
                if (modelEntity != gz::sim::kNullEntity) {
                    // Find the link entity that is a child of the model
                    this->linkEntity = _ecm.EntityByComponents(
                        gz::sim::components::Name(this->targetLinkName),
                        gz::sim::components::ParentEntity(modelEntity)
                    );
                }

                if (this->linkEntity == gz::sim::kNullEntity) return;
            }

            gz::sim::Link link(this->linkEntity);
            auto poseOpt = link.WorldPose(_ecm);
            if (!poseOpt) return;
            gz::math::Vector3d pos = poseOpt->Pos();

            gz::math::Vector3d total_force(0, 0, 0);

            for (const auto& zone : this->zones) {
                // Calculate distance from center to drone in each axis
                double dx = std::abs(pos.X() - zone.center.X());
                double dy = std::abs(pos.Y() - zone.center.Y());
                double dz = std::abs(pos.Z() - zone.center.Z());

                // Half-extents of the core
                double cx = zone.core_size.X() / 2.0;
                double cy = zone.core_size.Y() / 2.0;
                double cz = zone.core_size.Z() / 2.0;

                // Calculate distance outward from the core boundary
                double dist_x = std::max(0.0, dx - cx);
                double dist_y = std::max(0.0, dy - cy);
                double dist_z = std::max(0.0, dz - cz);

                // Maximum distance from core surface to drone
                double max_dist_from_core = std::max({dist_x, dist_y, dist_z});

                // If inside outer buffer bounds
                if (max_dist_from_core <= zone.buffer) {
                    double intensity = 1.0;
                    
                    // Linear falloff if inside the buffer but outside the core
                    if (max_dist_from_core > 0.0) {
                        intensity = 1.0 - (max_dist_from_core / zone.buffer);
                    }

                    double std_dev = std::max(0.0001, zone.noise_std_dev * intensity);
                    std::normal_distribution<double> dist(0.0, std_dev);
                    double noiseX = dist(gen);
                    double noiseY = dist(gen);
                    double noiseZ = dist(gen) * 0.5;

                    total_force += (zone.force * intensity) + gz::math::Vector3d(noiseX, noiseY, noiseZ);
                }
            }

            // Determine if we are currently inside a wind zone
            bool currently_in_zone = (total_force != gz::math::Vector3d::Zero);

            // Edge-triggered debug printing
            if (currently_in_zone && !this->was_in_zone) {
                gzmsg << "\n[Aviary Wind] -> ENTERED WIND COLUMN! Base force: " << total_force << std::endl;
            } else if (!currently_in_zone && this->was_in_zone) {
                gzmsg << "[Aviary Wind] <- EXITED WIND COLUMN.\n" << std::endl;
            }
            this->was_in_zone = currently_in_zone;

            // Apply the physics
            if (currently_in_zone) {
                gz::msgs::Wrench wrenchMsg;
                gz::msgs::Set(wrenchMsg.mutable_force(), total_force);
                gz::msgs::Set(wrenchMsg.mutable_torque(), gz::math::Vector3d::Zero);

                auto wrenchComp = _ecm.Component<gz::sim::components::ExternalWorldWrenchCmd>(this->linkEntity);
                if (wrenchComp) {
                    gz::math::Vector3d currentForce = gz::msgs::Convert(wrenchComp->Data().force());
                    gz::msgs::Set(wrenchComp->Data().mutable_force(), currentForce + total_force);
                } else {
                    _ecm.CreateComponent(this->linkEntity, gz::sim::components::ExternalWorldWrenchCmd(wrenchMsg));
                }
            }
        }

    private:
        std::string targetModelName = "x500_0";
        std::string targetLinkName = "base_link";
        gz::sim::Entity linkEntity{gz::sim::kNullEntity};
        std::vector<WindZone> zones;
        std::mt19937 gen;
        bool was_in_zone = false;
    };
}

GZ_ADD_PLUGIN(aviary::LocalizedWindPlugin, gz::sim::System, gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(aviary::LocalizedWindPlugin, "aviary::LocalizedWindPlugin")