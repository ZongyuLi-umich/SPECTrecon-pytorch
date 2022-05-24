using SPECTrecon

function gen_plan(mumap, psf, dy)
    return SPECTplan(mumap, psf, dy)
end

function gen_project(image, mumap, psf, dy)
    plan = gen_plan(mumap, psf, dy)
    return project(image, plan)
end

function gen_backproject(views, mumap, psf, dy)
    plan = gen_plan(mumap, psf, dy)
    return backproject(views, plan)
end
