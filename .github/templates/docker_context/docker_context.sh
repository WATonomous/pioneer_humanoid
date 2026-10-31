#!/bin/bash
set -e

################# Sweep for Docker Services and Modules #################
# Scans for services and modules in this repo,
# dynamically builds a json matrix for downstream CI build and testing

# Find docker compose files in 'modules' directory
modules=$(find ${GITHUB_WORKSPACE}/modules -maxdepth 1 -name "docker-compose*")

# Initialize an empty array for JSON objects
json_objects=()

# Check for infrastructure changes
TEST_ALL=false

if [[ $MODIFIED_MODULES = "infrastructure" ]]; then
    TEST_ALL=true
    echo Testing all "$MODULES_DIR"
fi

# Loop through each module
while read -r module; do
    module_out=$(basename "$(echo "$module" | sed -n 's/modules\/docker-compose\.\(.*\)\.yaml/\1/p')")

    # Skip GPU / non-CI modules
    # TODO: Add custom handling for embedded testing
    if [[ 'simulation' = $module_out || 'simulation_isaac' = $module_out || 'simulation_isaac_garment' = $module_out || 'simulation_mj' = $module_out || 'embedded' = $module_out ]]; then
        continue
    fi

    # Retrieve docker compose service names
    services=$(docker compose -f "$module" config --services)

    # Only work with modules that are modified
    if [[ $MODIFIED_MODULES != *$module_out* && $TEST_ALL = "false"  ]]; then
        continue
    fi

    # Loop through each service
    while read -r service_out; do
        # Temporarily skip perception services that have too large image size
        # if  [[ "$service_out" == "lane_detection" ]] || \
        #     [[ "$service_out" == "camera_object_detection" ]] || \
        #     continue
        # fi

        # TODO: Expose whole profile object to env
        dockerfile=$(docker compose -f "$module" config | yq ".services.$service_out.build.dockerfile")
        # Services that reuse another service's image (no build section) have nothing to build
        if [[ "$dockerfile" == "null" ]]; then
            continue
        fi
        json_object=$(jq -nc --arg module_out "$module_out" --arg service_out "$service_out" --arg dockerfile "$dockerfile" \
        '{module: $module_out, service: $service_out, dockerfile: $dockerfile}')
        # Append JSON object to the array
        json_objects+=($json_object)
        
    done <<< "$services"
done <<< "$modules"

# Convert the array of JSON objects to a single JSON array
json_services=$(jq -nc '[( $ARGS.positional[] | fromjson )]' --args -- ${json_objects[*]})

if [ -z $json_services ]; then
    exit 1
fi

echo output "docker_matrix=$(echo $json_services | jq -c '{include: .}')"
echo "docker_matrix=$(echo $json_services | jq -c '{include: .}')" >> $GITHUB_OUTPUT

################# Setup Docker Registry and Repository Name #################
# Docker Registry to pull/push images
REGISTRY_URL="ghcr.io/watonomous/humanoid"

REGISTRY=$(echo "$REGISTRY_URL" | sed 's|^\(.*\)/.*$|\1|')
REPOSITORY=$(echo "$REGISTRY_URL" | sed 's|^.*/\(.*\)$|\1|')

echo "registry=$REGISTRY" >> $GITHUB_OUTPUT
echo "repository=$REPOSITORY" >> $GITHUB_OUTPUT